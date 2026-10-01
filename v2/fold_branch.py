# fold_branch.py
"""Folded branch: encodes the light curve folded at every candidate period.

For each object and each candidate period P:
  1. every point gets a phase = (t / P) mod 1,
  2. the points (magnitude + band embedding) are encoded by a transformer whose attention uses
     rotary position embeddings on the phase angle (PhaseRoPEAttention). Rotations use integer
     harmonics of 2*pi*phase, so attention depends only on phase differences: the arbitrary
     phase zero point does not matter, and phase 0.99 is next to phase 0.01,
  3. attention pooling turns the points into one vector per fold.
The same weights encode every fold. The fold vectors, each with a descriptor of its period and
the number of cycles covered, are then compared by a small transformer across candidates,
with an extra object token whose output summarises the object. A linear head scores each
candidate; the highest score is the model's preferred period.

The `encoder="cnn"` option swaps the transformer for the circular-padding 1D CNN baseline in
cnn_encoder.py, keeping everything else identical.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from cnn_encoder import CircularCNNEncoder


class PhaseRoPEAttention(nn.Module):
    """Multi-head self-attention with rotary position embeddings on the phase angle.

    Each pair of query/key channels is rotated by h * 2*pi*phase, with the integer harmonic h
    cycling through 1..n_harm across channel pairs. Because the harmonics are integers, the
    rotation is periodic in phase, and the query-key product depends only on phase differences.
    """

    def __init__(self, d_model: int, n_heads: int, n_harm: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("The attention head dimension must be even")
        if n_harm < 1:
            raise ValueError("n_harm must be positive")

        self.n_heads = n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer(
            "harmonics",
            torch.arange(self.head_dim // 2).remainder(n_harm).add(1),
            persistent=False,
        )

    def _rotate(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Rotate consecutive channel pairs of x by harmonic * 2*pi*phase (complex rotation)."""
        angle = (
            phase[:, None, :, None].to(x.dtype)
            * self.harmonics.to(x.dtype)[None, None, None, :]
            * (2 * math.pi)
        )
        pairs = x.reshape(*x.shape[:-1], self.head_dim // 2, 2)
        real, imag = pairs.unbind(-1)
        cos, sin = angle.cos(), angle.sin()
        return torch.stack(
            (real * cos - imag * sin, real * sin + imag * cos), dim=-1
        ).flatten(-2)

    def forward(
        self, x: torch.Tensor, phase: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        folds, length, width = x.shape
        qkv = self.qkv(x).reshape(
            folds, length, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self._rotate(q, phase)
        k = self._rotate(k, phase)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(folds, length, width)
        return self.out(attended)


class _FoldBlock(nn.Module):
    """Pre-norm transformer block (phase-RoPE attention, then an MLP), with residuals."""

    def __init__(self, d_model: int, n_heads: int, n_harm: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhaseRoPEAttention(d_model, n_heads, n_harm, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, phase: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), phase, mask))
        return x + self.drop(self.mlp(self.norm2(x)))


class FoldEncoder(nn.Module):
    """Transformer over the points of one fold, followed by attention pooling to one vector.

    Pooling uses learned per-head queries; `pool_mask` lets hidden (masked) points take part
    in attention while being excluded from the pooled summary.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        n_harm: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.blocks = nn.ModuleList(
            _FoldBlock(d_model, n_heads, n_harm, dropout)
            for _ in range(n_layers)
        )
        self.pool_query = nn.Parameter(
            torch.randn(n_heads, self.head_dim) * 0.02
        )
        self.pool_key = nn.Linear(d_model, d_model)
        self.pool_value = nn.Linear(d_model, d_model)
        self.pool_out = nn.Linear(d_model, d_model)
        self.pool_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        points_tokens: torch.Tensor,
        phase: torch.Tensor,
        mask: torch.Tensor,
        return_weights: bool = False,
        pool_mask: torch.Tensor | None = None,
        return_tokens: bool = False,
    ):
        x = points_tokens
        for block in self.blocks:
            x = block(x, phase, mask)

        # Attention pooling: one learned query per head scores every point.
        folds, length, width = x.shape
        keys = self.pool_key(x).reshape(
            folds, length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        values = self.pool_value(x).reshape(
            folds, length, self.n_heads, self.head_dim
        ).transpose(1, 2)

        logits = (
            keys * self.pool_query[None, :, None, :]
        ).sum(-1) / math.sqrt(self.head_dim)
        if pool_mask is None:
            pool_mask = mask
        # Folds with no poolable point get a harmless uniform softmax, then zero weights.
        has_points = pool_mask.any(dim=-1, keepdim=True)
        safe_mask = pool_mask | ~has_points
        logits = logits.masked_fill(~safe_mask[:, None, :], -torch.inf)
        weights = logits.softmax(dim=-1)
        weights = weights * pool_mask[:, None, :].to(weights.dtype)

        pooled = (weights[..., None] * values).sum(dim=2)
        pooled = pooled.reshape(folds, width)
        fold_vector = self.pool_norm(self.pool_out(pooled))

        if return_weights and return_tokens:
            return fold_vector, weights.mean(dim=1), x
        if return_weights:
            return fold_vector, weights.mean(dim=1)
        if return_tokens:
            return fold_vector, x
        return fold_vector


class _CandidateBlock(nn.Module):
    """Pre-norm transformer block across candidate periods (no positional encoding: the
    candidates form a set)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        y = self.norm1(x)
        qkv = self.qkv(y).reshape(
            batch, length, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(batch, length, width)
        x = x + self.drop(self.out(y))
        return x + self.drop(self.mlp(self.norm2(x)))


class FoldBranch(nn.Module):
    """Fold every candidate period, encode each fold, then compare the candidates.

    Returns a dict with:
      z           object summary vector (output of the object token),
      cand_scores one score per candidate period (-inf for padding),
      cand_vecs   one vector per candidate,
      fold_pred   per-point magnitude predictions per fold (only when `point_head` is on and
                  a `hidden` mask is given, i.e. during masked pretraining).
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        n_cand_layers: int = 2,
        n_harm: int = 4,
        dropout: float = 0.1,
        fold_chunk: int = 64,
        grad_checkpoint: bool = False,
        encoder: str = "rope",
        cnn_kwargs: dict | None = None,
        point_head: bool = False,
        side_dim: int = 0,
    ):
        super().__init__()
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer")
        if fold_chunk < 1:
            raise ValueError("fold_chunk must be positive")
        if encoder not in ("rope", "cnn"):
            raise ValueError("encoder must be 'rope' or 'cnn'")

        self.d_model = d_model
        self.fold_chunk = fold_chunk
        self.grad_checkpoint = grad_checkpoint
        self.encoder = encoder
        self.point_head_enabled = point_head
        self.side_dim = side_dim

        if encoder == "rope":
            self.mag_proj = nn.Linear(1, d_model)
            self.mask_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
            self.band_embed = nn.Embedding(6, d_model)
            self.point_norm = nn.LayerNorm(d_model)
            self.fold_encoder = FoldEncoder(
                d_model, n_heads, n_layers, n_harm, dropout
            )
            if point_head:
                self.point_head = nn.Linear(d_model, 1)
        else:
            options = {"d_model": d_model, "dropout": dropout}
            options.update(cnn_kwargs or {})
            if options["d_model"] != d_model:
                raise ValueError("cnn_kwargs d_model must match FoldBranch d_model")
            self.fold_encoder = CircularCNNEncoder(**options)
            if point_head:
                self.pred_band_embed = nn.Embedding(
                    self.fold_encoder.n_bands, d_model
                )
                self.point_head = nn.Linear(
                    self.fold_encoder.feature_width + d_model, 1
                )

        # Maps (log10 period, log10 cycles covered) to a vector added to each fold vector.
        self.descriptor = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.object_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.candidate_blocks = nn.ModuleList(
            _CandidateBlock(d_model, n_heads, dropout)
            for _ in range(n_cand_layers)
        )
        self.candidate_norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)
        if side_dim > 0:
            if encoder == "rope":
                self.side_point = nn.Linear(side_dim, d_model)
            self.side_desc = nn.Linear(side_dim, d_model)

    @staticmethod
    def compute_phase(t: torch.Tensor, periods: torch.Tensor) -> torch.Tensor:
        """Return phases with shape [B, K, N]."""
        return torch.remainder(
            t.to(torch.float64)[:, None, :]
            / periods.to(torch.float64)[:, :, None],
            1.0,
        ).to(torch.float32)

    def forward(self, batch: dict) -> dict:
        t = batch["t"]
        periods = batch["periods"]
        period_mask = batch["period_mask"]
        point_mask = batch["point_mask"]
        side = batch.get("side")
        hidden = batch.get("hidden")
        if hidden is not None:
            hidden = hidden & point_mask
        predict = self.point_head_enabled and hidden is not None

        batch_size, n_points = t.shape
        n_candidates = periods.shape[1]

        # Point tokens: magnitude projection + band embedding. Hidden points have their
        # magnitude replaced by a learned mask embedding, so the model cannot see them.
        if self.encoder == "rope":
            if hidden is None:
                magnitude_tokens = self.mag_proj(batch["mag"].unsqueeze(-1))
            else:
                safe_mag = batch["mag"].masked_fill(hidden, 0.0)
                magnitude_tokens = self.mag_proj(safe_mag.unsqueeze(-1))
                magnitude_tokens = torch.where(
                    hidden[..., None],
                    self.mask_embedding,
                    magnitude_tokens,
                )
            point_tokens = magnitude_tokens + self.band_embed(batch["band"])
            if self.side_dim > 0 and side is not None:
                point_tokens = point_tokens + self.side_point(side)[:, None, :]
            points = self.point_norm(point_tokens)

        # Encode all real (object, candidate) folds, `fold_chunk` folds at a time to bound
        # memory. Phases are computed in float64 because t / P can be large for short periods.
        valid = period_mask.reshape(-1).nonzero(as_tuple=True)[0]
        fold_vectors = t.new_zeros((batch_size * n_candidates, self.d_model))
        if predict:
            predictions = t.new_zeros((batch_size * n_candidates, n_points))
        flat_periods = periods.reshape(-1)

        for indices in valid.split(self.fold_chunk):
            objects = torch.div(indices, n_candidates, rounding_mode="floor")
            phase = torch.remainder(
                t.index_select(0, objects).to(torch.float64)
                / flat_periods.index_select(0, indices).to(torch.float64)[:, None],
                1.0,
            ).to(torch.float32)
            fold_mask = point_mask.index_select(0, objects)

            if self.encoder == "rope":
                args = (points.index_select(0, objects), phase, fold_mask)
                kwargs = {}
                if hidden is not None:
                    kwargs["pool_mask"] = fold_mask & ~hidden.index_select(
                        0, objects
                    )
                if predict:
                    kwargs["return_tokens"] = True
            else:
                visible = fold_mask
                if hidden is not None:
                    visible = fold_mask & ~hidden.index_select(0, objects)
                args = (
                    batch["mag"].index_select(0, objects),
                    batch["band"].index_select(0, objects),
                    phase,
                    visible,
                )
                kwargs = {"return_features": True} if predict else {}

            if self.grad_checkpoint and self.training:
                encoded = checkpoint(
                    self.fold_encoder, *args, use_reentrant=False, **kwargs
                )
            else:
                encoded = self.fold_encoder(*args, **kwargs)

            # During masked pretraining, also predict each point's magnitude from its fold.
            if predict:
                vectors, features = encoded
                if self.encoder == "rope":
                    point_predictions = self.point_head(features).squeeze(-1)
                else:
                    bins = torch.floor(
                        phase * self.fold_encoder.n_bins
                    ).long().clamp(0, self.fold_encoder.n_bins - 1)
                    feature_length = features.shape[-1]
                    feature_bins = torch.div(
                        bins * feature_length,
                        self.fold_encoder.n_bins,
                        rounding_mode="floor",
                    ).remainder(feature_length)
                    gathered = features.gather(
                        2,
                        feature_bins[:, None, :].expand(
                            -1, features.shape[1], -1
                        ),
                    ).transpose(1, 2)
                    bands = self.pred_band_embed(
                        batch["band"].index_select(0, objects)
                    )
                    point_predictions = self.point_head(
                        torch.cat((gathered, bands), dim=-1)
                    ).squeeze(-1)

                point_predictions = point_predictions.masked_fill(
                    ~fold_mask, 0.0
                )
                predictions = predictions.index_copy(
                    0, indices, point_predictions.to(predictions.dtype)
                )
            else:
                vectors = encoded

            fold_vectors = fold_vectors.index_copy(0, indices, vectors.to(fold_vectors.dtype))

        # Add the period descriptor, then let the candidates attend to each other together with
        # the object token (position 0).
        descriptors = torch.stack(
            (
                periods.clamp_min(1e-12).log10(),
                batch["cycles"].clamp_min(1e-3).log10(),
            ),
            dim=-1,
        )
        candidates = (
            fold_vectors.reshape(batch_size, n_candidates, self.d_model)
            + self.descriptor(descriptors)
        )
        if self.side_dim > 0 and side is not None:
            candidates = candidates + self.side_desc(side)[:, None, :]
        candidates = candidates.masked_fill(~period_mask[..., None], 0.0)

        x = torch.cat(
            (self.object_cls.expand(batch_size, -1, -1), candidates), dim=1
        )
        mask = torch.cat(
            (
                torch.ones(
                    (batch_size, 1), dtype=torch.bool, device=period_mask.device
                ),
                period_mask,
            ),
            dim=1,
        )
        for block in self.candidate_blocks:
            x = block(x, mask)
        x = self.candidate_norm(x)

        z = x[:, 0]
        cand_vecs = x[:, 1:].masked_fill(~period_mask[..., None], 0.0)
        cand_scores = self.score(cand_vecs).squeeze(-1)
        cand_scores = cand_scores.masked_fill(~period_mask, -torch.inf)
        output = {
            "z": z,
            "cand_scores": cand_scores,
            "cand_vecs": cand_vecs,
        }
        if predict:
            output["fold_pred"] = predictions.reshape(
                batch_size, n_candidates, n_points
            )
        return output
