# pretrain.py
"""Self-supervised pretraining model: both branches plus the pretraining losses.

A fraction of each object's observing nights is hidden (masking.night_mask). The model must
predict the hidden magnitudes:
  * loss_unf:  the unfolded branch predicts each hidden point directly (L1 for Laplace).
  * loss_fold: every fold predicts the hidden points; the candidate scores act as mixture
               weights, so the loss rewards putting weight on folds that predict well.
               It is the negative log-likelihood of a mixture over candidates.
  * loss_fvu:  (optional) mean over candidates of each fold's fraction of variance
               unexplained (FVU) on the hidden points, capped at fvu_cap. Trains every fold to
               be predictive, not only the currently preferred ones. fvu_kind "sq" uses squared
               errors (v1), "abs" absolute errors (v2).
  * loss_rank: (optional) cross-entropy between the candidate scores and a target
               distribution softmax(-quality / rank_tau), which directly teaches the scores to
               rank candidates by how well their fold predicts unseen nights. quality = FVU, plus
               ce_weight times the cross-validated conditional entropy ratio (v2): a phase x
               magnitude histogram built from the VISIBLE points of each fold, scored on the
               HIDDEN points, relative to ignoring phase (~1 = no information, lower = better).
Total loss = loss_fold + loss_unf + fvu_weight * loss_fvu + rank_weight * loss_rank.

v1 (full50): Laplace likelihood, fvu_weight = rank_weight = 1, rank_tau = 0.1, fvu_cap = 5,
squared FVU, magnitudes pre-scaled in the cache.
v2: the same plus normalize=True (each object scaled by the spread of its visible points after
masking, see dataset.normalize_batch), side_dim=11 (colours and log amplitude into both
branches), fvu_kind="abs" and ce_weight=1.
"""
import math

import torch
from torch import nn

from dataset import normalize_batch
from fold_branch import FoldBranch
from unfolded_branch import UnfoldedBranch


# Cross-validated conditional entropy: histograms from visible points, scored on hidden ones.
def _conditional_entropy_counts(
    magnitude_bins, phase_indices, visible, phase_bins, mag_bins
):
    batch_size, n_candidates, _ = phase_indices.shape
    cell_indices = (
        phase_indices * mag_bins + magnitude_bins[:, None, :]
    )
    counts = torch.zeros(
        (batch_size, n_candidates, phase_bins * mag_bins),
        dtype=torch.float32,
        device=magnitude_bins.device,
    )
    counts.scatter_add_(
        2,
        cell_indices,
        visible[:, None, :].expand(-1, n_candidates, -1).float(),
    )
    marginal_counts = torch.zeros(
        (batch_size, mag_bins),
        dtype=torch.float32,
        device=magnitude_bins.device,
    )
    marginal_counts.scatter_add_(1, magnitude_bins, visible.float())
    return counts.reshape(
        batch_size, n_candidates, phase_bins, mag_bins
    ), marginal_counts


@torch.no_grad()
def conditional_entropy_ratio(
    mag,
    t,
    periods,
    period_mask,
    point_mask,
    hidden,
    phase_bins,
    mag_bins,
    alpha,
):
    """Score hidden points using conditional probabilities fitted on visible points."""
    if phase_bins < 2 or mag_bins < 2:
        raise ValueError("phase_bins and mag_bins must be at least 2")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    with torch.autocast(device_type=mag.device.type, enabled=False):
        point_mask = point_mask.bool()
        hidden = hidden.bool() & point_mask
        period_mask = period_mask.bool()
        visible = point_mask & ~hidden
        mag = mag.float().masked_fill(~point_mask, 0.0)
        n_visible = visible.sum(dim=-1)
        n_hidden = hidden.sum(dim=-1)

        quantiles = torch.linspace(
            0.0,
            1.0,
            mag_bins + 1,
            dtype=torch.float32,
            device=mag.device,
        )[1:-1]
        magnitude_bins = torch.zeros_like(mag, dtype=torch.int64)
        for row in range(mag.shape[0]):
            visible_mag = mag[row, visible[row]]
            if visible_mag.numel() >= 2:
                edges = torch.quantile(visible_mag, quantiles)
                magnitude_bins[row] = torch.bucketize(
                    mag[row].contiguous(), edges.contiguous()
                )

        safe_periods = periods.masked_fill(~period_mask, 1.0)
        phase = FoldBranch.compute_phase(
            t.masked_fill(~point_mask, 0.0), safe_periods
        )
        phase_indices = torch.floor(
            phase * phase_bins
        ).long().clamp(0, phase_bins - 1)

        counts, marginal_counts = _conditional_entropy_counts(
            magnitude_bins, phase_indices, visible, phase_bins, mag_bins
        )
        conditional_log_prob = (
            (counts + alpha).log()
            - (
                counts.sum(dim=-1, keepdim=True) + alpha * mag_bins
            ).log()
        )
        marginal_log_prob = (
            (marginal_counts + alpha).log()
            - (n_visible.float()[:, None] + alpha * mag_bins).log()
        )

        cell_indices = (
            phase_indices * mag_bins + magnitude_bins[:, None, :]
        )
        point_log_prob = conditional_log_prob.flatten(2).gather(
            2, cell_indices
        )
        hidden_counts = n_hidden.clamp_min(1).float()
        ce_cv = -point_log_prob.masked_fill(
            ~hidden[:, None, :], 0.0
        ).sum(dim=-1) / hidden_counts[:, None]
        h_cv = -marginal_log_prob.gather(
            1, magnitude_bins
        ).masked_fill(~hidden, 0.0).sum(dim=-1) / hidden_counts

        ratio = ce_cv / h_cv[:, None].clamp_min(1e-3)
        usable = (n_visible >= 2) & (n_hidden > 0)
        ratio = torch.where(
            usable[:, None] & period_mask,
            ratio,
            torch.ones_like(ratio),
        )
        return ratio.detach().float()


class PretrainModel(nn.Module):
    """FoldBranch + UnfoldedBranch with the masked-prediction losses described above."""

    def __init__(
        self,
        encoder: str = "rope",
        d_model: int = 64,
        fold_kwargs: dict | None = None,
        unfolded_kwargs: dict | None = None,
        likelihood: str = "laplace",
        fvu_weight: float = 0.0,
        rank_weight: float = 0.0,
        rank_tau: float = 0.1,
        fvu_cap: float = 5.0,
        side_dim: int = 0,
        fvu_kind: str = "sq",
        normalize: bool = False,
        ce_weight: float = 0.0,
        ce_phase_bins: int = 10,
        ce_mag_bins: int = 5,
        ce_alpha: float = 0.5,
    ):
        super().__init__()
        if likelihood not in ("laplace", "gaussian"):
            raise ValueError("likelihood must be 'laplace' or 'gaussian'")
        if fvu_weight < 0 or rank_weight < 0:
            raise ValueError("fvu_weight and rank_weight must be nonnegative")
        if rank_tau <= 0 or fvu_cap <= 0:
            raise ValueError("rank_tau and fvu_cap must be positive")
        if fvu_kind not in ("sq", "abs"):
            raise ValueError("fvu_kind must be 'sq' or 'abs'")
        if ce_weight < 0:
            raise ValueError("ce_weight must be nonnegative")
        if ce_phase_bins < 2 or ce_mag_bins < 2:
            raise ValueError("ce_phase_bins and ce_mag_bins must be at least 2")
        if ce_alpha <= 0:
            raise ValueError("ce_alpha must be positive")

        fold_options = dict(fold_kwargs or {})
        unfolded_options = dict(unfolded_kwargs or {})
        fold_options.update(
            d_model=d_model, encoder=encoder, point_head=True, side_dim=side_dim
        )
        unfolded_options.update(d_model=d_model, side_dim=side_dim)

        self.likelihood = likelihood
        self.fvu_weight = fvu_weight
        self.rank_weight = rank_weight
        self.rank_tau = rank_tau
        self.fvu_cap = fvu_cap
        self.fvu_kind = fvu_kind
        self.normalize = normalize
        self.ce_weight = ce_weight
        self.ce_phase_bins = ce_phase_bins
        self.ce_mag_bins = ce_mag_bins
        self.ce_alpha = ce_alpha

        self.fold = FoldBranch(**fold_options)
        self.unfolded = UnfoldedBranch(**unfolded_options)
        self.unfolded_point_head = nn.Linear(d_model, 1)
        if likelihood == "laplace":
            self.log_b = nn.Parameter(torch.tensor(math.log(0.5)))
        else:
            self.log_sigma = nn.Parameter(torch.tensor(math.log(0.5)))

    @staticmethod
    def fold_mixture(
        fold_pred: torch.Tensor,
        cand_scores: torch.Tensor,
        target: torch.Tensor,
        hidden: torch.Tensor,
        period_mask: torch.Tensor,
        log_sigma: torch.Tensor,
        likelihood: str = "gaussian",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the candidate mixture and detached selection diagnostics.

        loss_fold = -log sum_k softmax(scores)_k * p(hidden points | fold k), averaged per
        hidden point. Diagnostics (no gradient) report the score entropy and whether the
        top-scored fold is also the fold with the lowest error.
        """
        if likelihood not in ("laplace", "gaussian"):
            raise ValueError("likelihood must be 'laplace' or 'gaussian'")

        pred = fold_pred.float()
        scores = cand_scores.float()
        target = target.float()
        hidden = hidden.bool()
        period_mask = period_mask.bool()

        n_hidden = hidden.sum(dim=-1)
        eligible = (n_hidden > 0) & period_mask.any(dim=-1)
        counts = n_hidden.clamp_min(1).float()
        valid_scores = scores.masked_fill(~period_mask, -torch.inf)
        safe_scores = torch.where(
            period_mask.any(dim=-1, keepdim=True),
            valid_scores,
            torch.zeros_like(valid_scores),
        )
        log_weights = torch.log_softmax(safe_scores, dim=-1)
        log_weights = log_weights.masked_fill(~period_mask, -torch.inf)

        residual = pred - target[:, None, :]
        squared = residual.square() * hidden[:, None, :]
        sum_squared = squared.sum(dim=-1)
        mean_squared = sum_squared / counts[:, None]
        safe_log_scale = log_sigma.float().clamp(-8.0, 5.0)

        if likelihood == "laplace":
            sum_absolute = (
                residual.abs() * hidden[:, None, :]
            ).sum(dim=-1)
            log_likelihood = (
                -sum_absolute * torch.exp(-safe_log_scale)
                - counts[:, None] * (math.log(2.0) + safe_log_scale)
            )
        else:
            inv_variance = torch.exp(-2.0 * safe_log_scale)
            log_likelihood = (
                -0.5 * sum_squared * inv_variance
                - counts[:, None]
                * (safe_log_scale + 0.5 * math.log(2.0 * math.pi))
            )

        mixture_log_likelihood = torch.logsumexp(
            log_weights + log_likelihood, dim=-1
        )
        if eligible.any():
            loss_fold = (
                -mixture_log_likelihood[eligible] / counts[eligible]
            ).mean()
        else:
            loss_fold = fold_pred.float().sum() * 0.0

        with torch.no_grad():
            if eligible.any():
                selected_scores = valid_scores[eligible]
                selected_mse = mean_squared[eligible].masked_fill(
                    ~period_mask[eligible], torch.inf
                )
                top = selected_scores.argmax(dim=-1)
                best = selected_mse.argmin(dim=-1)
                rows = torch.arange(top.numel(), device=top.device)

                log_prob = torch.log_softmax(selected_scores, dim=-1)
                prob = log_prob.exp()
                entropy = -(
                    prob * log_prob.masked_fill(
                        ~period_mask[eligible], 0.0
                    )
                ).sum(dim=-1)

                diagnostics = {
                    "cand_entropy": entropy.mean(),
                    "top_is_best": (top == best).float().mean(),
                    "mse_best_fold": selected_mse[rows, best].mean(),
                    "mse_top_fold": selected_mse[rows, top].mean(),
                }
            else:
                zero = target.new_zeros(())
                diagnostics = {
                    "cand_entropy": zero,
                    "top_is_best": zero,
                    "mse_best_fold": zero,
                    "mse_top_fold": zero,
                }

        return loss_fold, diagnostics

    @staticmethod
    def fold_quality(
        fold_pred: torch.Tensor,
        cand_scores: torch.Tensor,
        target: torch.Tensor,
        hidden: torch.Tensor,
        period_mask: torch.Tensor,
        rank_tau: float = 0.1,
        fvu_cap: float = 5.0,
        fvu_kind: str = "sq",
        ce_ratio: torch.Tensor | None = None,
        ce_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Cross-validated FVU losses and detached diagnostics.

        "Cross-validated" because FVU is measured only on hidden points, which no fold saw.
        "sq": FVU = sum of squared errors / sum of squared deviations from the hidden points' mean.
        "abs": FVU = sum of absolute errors / sum of absolute deviations from their median.
        FVU < 1 means the fold predicts better than a flat line. With ce_weight > 0 the ranking
        target also uses ce_ratio (see conditional_entropy_ratio).
        """
        if rank_tau <= 0 or fvu_cap <= 0:
            raise ValueError("rank_tau and fvu_cap must be positive")
        if fvu_kind not in ("sq", "abs"):
            raise ValueError("fvu_kind must be 'sq' or 'abs'")
        if ce_weight < 0:
            raise ValueError("ce_weight must be nonnegative")

        pred = fold_pred.float()
        scores = cand_scores.float()
        y = target.float()
        hidden = hidden.bool()
        period_mask = period_mask.bool()

        n_hidden = hidden.sum(dim=-1)
        eligible = (n_hidden >= 3) & period_mask.any(dim=-1)

        if not eligible.any():
            loss_fvu = pred.sum() * 0.0
            loss_rank = scores.masked_fill(
                ~period_mask, 0.0
            ).sum() * 0.0
            zero = y.new_zeros(())
            return loss_fvu, loss_rank, {
                "fvu_top_median": zero,
                "fvu_best_median": zero,
                "frac_top_fvu_lt1": zero,
                "frac_top_fvu_lt03": zero,
                "rank_target_entropy": zero,
                "loss_rank_kl": zero,
                "ce_ratio_top_median": zero,
                "ce_ratio_best_median": zero,
            }

        pred = pred[eligible]
        scores = scores[eligible]
        y = y[eligible]
        hidden = hidden[eligible]
        period_mask = period_mask[eligible]
        counts = n_hidden[eligible].float()

        if fvu_kind == "sq":
            ybar = (
                (y * hidden).sum(dim=-1) / counts
            )
            centered = (y - ybar[:, None]) * hidden
            denominator = centered.square().sum(dim=-1).clamp_min(
                counts * 0.01
            )
            squared_error = (
                (pred - y[:, None, :]).square()
                * hidden[:, None, :]
            ).sum(dim=-1)
            fvu = squared_error / denominator[:, None]
        else:
            median = torch.nanmedian(
                y.masked_fill(~hidden, torch.nan), dim=-1
            ).values
            median = torch.where(
                hidden.any(dim=-1), median, torch.zeros_like(median)
            )
            denominator = (
                (y - median[:, None]).abs() * hidden
            ).sum(dim=-1).clamp_min(counts * 0.1)
            absolute_error = (
                (pred - y[:, None, :]).abs()
                * hidden[:, None, :]
            ).sum(dim=-1)
            fvu = absolute_error / denominator[:, None]

        valid_count = period_mask.sum(dim=-1).float()
        capped = fvu.clamp(max=fvu_cap)
        loss_fvu = (
            (capped * period_mask).sum(dim=-1) / valid_count
        ).mean()

        # Ranking target: softmax(-FVU / rank_tau), with FVU detached so it is a fixed target.
        valid_scores = scores.masked_fill(~period_mask, -torch.inf)
        ranking_quality = fvu.detach()
        if ce_weight > 0:
            if ce_ratio is None:
                raise ValueError("ce_ratio is required when ce_weight is positive")
            ce = ce_ratio.detach().float()[eligible]
            ranking_quality = ranking_quality + ce_weight * ce
        target_logits = (
            -ranking_quality / rank_tau
        ).masked_fill(~period_mask, -torch.inf)
        q = torch.softmax(target_logits, dim=-1)
        log_prob = torch.log_softmax(valid_scores, dim=-1)
        loss_rank = -(
            q * log_prob.masked_fill(~period_mask, 0.0)
        ).sum(dim=-1).mean()

        with torch.no_grad():
            rank_target_entropy = -(
                q * q.log().masked_fill(
                    (q == 0) | ~period_mask, 0.0
                )
            ).sum(dim=-1).mean()
            top = valid_scores.argmax(dim=-1)
            best = fvu.masked_fill(
                ~period_mask, torch.inf
            ).min(dim=-1).values
            rows = torch.arange(top.numel(), device=top.device)
            top_fvu = fvu[rows, top]
            zero = y.new_zeros(())
            ce_top = zero
            ce_best = zero
            if ce_weight > 0:
                ce_top = ce[rows, top].median()
                ce_best = ce.masked_fill(
                    ~period_mask, torch.inf
                ).min(dim=-1).values.median()
            diagnostics = {
                "fvu_top_median": top_fvu.median(),
                "fvu_best_median": best.median(),
                "frac_top_fvu_lt1": (top_fvu < 1.0).float().mean(),
                "frac_top_fvu_lt03": (top_fvu < 0.3).float().mean(),
                "rank_target_entropy": rank_target_entropy.detach(),
                "loss_rank_kl": (
                    loss_rank.detach() - rank_target_entropy
                ).detach(),
                "ce_ratio_top_median": ce_top.detach(),
                "ce_ratio_best_median": ce_best.detach(),
            }

        return loss_fvu, loss_rank, diagnostics

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "hidden" not in batch:
            raise KeyError("Pretraining requires batch['hidden']")

        hidden = batch["hidden"].bool() & batch["point_mask"].bool()
        if self.normalize:
            batch = normalize_batch(batch, hidden)
        model_batch = dict(batch, hidden=hidden)
        fold_output = self.fold(model_batch)
        unfolded_output = self.unfolded(model_batch)

        # Unfolded branch: predict each hidden magnitude from its token.
        target = batch["mag"].float()
        unfolded_pred = self.unfolded_point_head(
            unfolded_output["tokens"]
        ).squeeze(-1).float()
        if hidden.any():
            residual = unfolded_pred[hidden] - target[hidden]
            if self.likelihood == "laplace":
                loss_unf = residual.abs().mean()
            else:
                loss_unf = residual.square().mean()
        else:
            loss_unf = unfolded_pred.sum() * 0.0

        log_scale = (
            self.log_b if self.likelihood == "laplace" else self.log_sigma
        )
        loss_fold, mixture_diagnostics = self.fold_mixture(
            fold_output["fold_pred"],
            fold_output["cand_scores"],
            target,
            hidden,
            batch["period_mask"],
            log_scale,
            likelihood=self.likelihood,
        )
        ce_ratio = None
        if self.ce_weight > 0:
            ce_ratio = conditional_entropy_ratio(
                target,
                batch["t"],
                batch["periods"],
                batch["period_mask"],
                batch["point_mask"],
                hidden,
                self.ce_phase_bins,
                self.ce_mag_bins,
                self.ce_alpha,
            )
        loss_fvu, loss_rank, fvu_diagnostics = self.fold_quality(
            fold_output["fold_pred"],
            fold_output["cand_scores"],
            target,
            hidden,
            batch["period_mask"],
            rank_tau=self.rank_tau,
            fvu_cap=self.fvu_cap,
            fvu_kind=self.fvu_kind,
            ce_ratio=ce_ratio,
            ce_weight=self.ce_weight,
        )

        # Keep the original addition order when both new weights are zero.
        loss = loss_fold + loss_unf
        if self.fvu_weight != 0.0:
            loss = loss + self.fvu_weight * loss_fvu
        if self.rank_weight != 0.0:
            loss = loss + self.rank_weight * loss_rank

        return {
            "loss": loss,
            "loss_fold": loss_fold,
            "loss_unf": loss_unf,
            "loss_fvu": loss_fvu,
            "loss_rank": loss_rank,
            "sigma": log_scale.detach().float().clamp(-8.0, 5.0).exp(),
            **mixture_diagnostics,
            **fvu_diagnostics,
        }
