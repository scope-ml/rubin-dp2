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
               be predictive, not only the currently preferred ones.
  * loss_rank: (optional) cross-entropy between the candidate scores and a target
               distribution softmax(-FVU / rank_tau), which directly teaches the scores to rank
               candidates by how well their fold predicts unseen nights.
Total loss = loss_fold + loss_unf + fvu_weight * loss_fvu + rank_weight * loss_rank.
The full run used a Laplace likelihood with fvu_weight = rank_weight = 1, rank_tau = 0.1 and
fvu_cap = 5.
"""
import math

import torch
from torch import nn

from fold_branch import FoldBranch
from unfolded_branch import UnfoldedBranch


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
    ):
        super().__init__()
        if likelihood not in ("laplace", "gaussian"):
            raise ValueError("likelihood must be 'laplace' or 'gaussian'")
        if fvu_weight < 0 or rank_weight < 0:
            raise ValueError("fvu_weight and rank_weight must be nonnegative")
        if rank_tau <= 0 or fvu_cap <= 0:
            raise ValueError("rank_tau and fvu_cap must be positive")

        fold_options = dict(fold_kwargs or {})
        unfolded_options = dict(unfolded_kwargs or {})
        fold_options.update(d_model=d_model, encoder=encoder, point_head=True)
        unfolded_options["d_model"] = d_model

        self.likelihood = likelihood
        self.fvu_weight = fvu_weight
        self.rank_weight = rank_weight
        self.rank_tau = rank_tau
        self.fvu_cap = fvu_cap

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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Cross-validated FVU losses and detached diagnostics.

        "Cross-validated" because FVU is measured only on hidden points, which no fold saw.
        FVU = sum of squared errors / sum of squared deviations from the hidden points' mean;
        FVU < 1 means the fold predicts better than a flat line.
        """
        if rank_tau <= 0 or fvu_cap <= 0:
            raise ValueError("rank_tau and fvu_cap must be positive")

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
            }

        pred = pred[eligible]
        scores = scores[eligible]
        y = y[eligible]
        hidden = hidden[eligible]
        period_mask = period_mask[eligible]
        counts = n_hidden[eligible].float()

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

        valid_count = period_mask.sum(dim=-1).float()
        capped = fvu.clamp(max=fvu_cap)
        loss_fvu = (
            (capped * period_mask).sum(dim=-1) / valid_count
        ).mean()

        # Ranking target: softmax(-FVU / rank_tau), with FVU detached so it is a fixed target.
        valid_scores = scores.masked_fill(~period_mask, -torch.inf)
        target_logits = (
            -fvu.detach() / rank_tau
        ).masked_fill(~period_mask, -torch.inf)
        q = torch.softmax(target_logits, dim=-1)
        log_prob = torch.log_softmax(valid_scores, dim=-1)
        loss_rank = -(
            q * log_prob.masked_fill(~period_mask, 0.0)
        ).sum(dim=-1).mean()

        with torch.no_grad():
            top = valid_scores.argmax(dim=-1)
            best = fvu.masked_fill(
                ~period_mask, torch.inf
            ).min(dim=-1).values
            rows = torch.arange(top.numel(), device=top.device)
            top_fvu = fvu[rows, top]
            diagnostics = {
                "fvu_top_median": top_fvu.median(),
                "fvu_best_median": best.median(),
                "frac_top_fvu_lt1": (top_fvu < 1.0).float().mean(),
                "frac_top_fvu_lt03": (top_fvu < 0.3).float().mean(),
            }

        return loss_fvu, loss_rank, diagnostics

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "hidden" not in batch:
            raise KeyError("Pretraining requires batch['hidden']")

        hidden = batch["hidden"].bool() & batch["point_mask"].bool()
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
        loss_fvu, loss_rank, fvu_diagnostics = self.fold_quality(
            fold_output["fold_pred"],
            fold_output["cand_scores"],
            target,
            hidden,
            batch["period_mask"],
            rank_tau=self.rank_tau,
            fvu_cap=self.fvu_cap,
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
