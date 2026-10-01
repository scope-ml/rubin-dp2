# unfolded_branch.py
"""Unfolded branch: a transformer over the raw, time-ordered light curve.

Each point is a token: magnitude projection + band embedding + a sinusoidal time embedding at
fixed, log-spaced periods (t_min = 0.005 d to t_max = 4000 d). Attention pooling gives one
vector per object, to which a small MLP adds log10(baseline) and log10(number of points).
This branch sees trends, outbursts and aperiodic behaviour that folding would scramble.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class TimeEmbedding(nn.Module):
    """sin/cos of t at fixed log-spaced periods, linearly projected to d_model."""

    def __init__(self, d_model: int, n_time_freqs: int, t_min: float, t_max: float):
        super().__init__()
        if n_time_freqs < 1 or t_min <= 0 or t_max < t_min:
            raise ValueError("Require n_time_freqs >= 1 and 0 < t_min <= t_max")

        periods = torch.logspace(
            math.log10(t_min),
            math.log10(t_max),
            n_time_freqs,
            dtype=torch.float64,
        )
        self.register_buffer("angular_frequencies", 2.0 * math.pi / periods)
        self.projection = nn.Linear(2 * n_time_freqs, d_model)

    def features(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.to(torch.float64).unsqueeze(-1) * self.angular_frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1).to(t.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(t))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with a padding mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.residual_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_points, _ = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1)

        def split_heads(y: torch.Tensor) -> torch.Tensor:
            return y.reshape(
                batch_size, n_points, self.n_heads, self.head_dim
            ).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            split_heads(q),
            split_heads(k),
            split_heads(v),
            attn_mask=point_mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, n_points, -1)
        x = x + self.residual_dropout(self.attn_out(attended))
        return x + self.mlp(self.mlp_norm(x))


class UnfoldedBranch(nn.Module):
    """Encode the unfolded light curve. Returns the object vector `z`, the per-point `tokens`
    (used to predict hidden magnitudes during pretraining) and the pooling weights."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        n_time_freqs: int = 32,
        t_min: float = 0.005,
        t_max: float = 300.0,  # longest period (days); DP2 baselines reach 238 d
        dropout: float = 0.1,
        grad_checkpoint: bool = False,
        side_dim: int = 0,
    ):
        super().__init__()
        if d_model < 1 or n_heads < 1 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if n_layers < 0:
            raise ValueError("n_layers must be nonnegative")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.grad_checkpoint = grad_checkpoint
        self.side_dim = side_dim

        self.mag_projection = nn.Linear(1, d_model)
        self.mask_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        self.band_embedding = nn.Embedding(6, d_model)
        self.time_embedding = TimeEmbedding(
            d_model, n_time_freqs, t_min, t_max
        )
        self.token_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        )

        self.pool_query = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.normal_(self.pool_query, std=self.head_dim**-0.5)
        self.pool_keys = nn.Linear(d_model, d_model)
        self.pool_out = nn.Linear(d_model, d_model)
        self.pool_norm = nn.LayerNorm(d_model)
        self.metadata_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        if side_dim > 0:
            self.side_token = nn.Linear(side_dim, d_model)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        t = batch["t"]
        band = batch["band"]
        mag = batch["mag"]
        point_mask = batch["point_mask"]
        baseline = batch["baseline"]
        side = batch.get("side")
        hidden = batch.get("hidden")

        if not torch.all(point_mask.any(dim=1)):
            raise ValueError("Each object must contain at least one real point")

        # Hidden points: magnitude replaced by a learned mask embedding, and excluded from
        # pooling. Their time and band stay visible, so the model knows where to predict.
        if hidden is None:
            magnitude_tokens = self.mag_projection(mag.unsqueeze(-1))
            pool_mask = point_mask
        else:
            hidden = hidden & point_mask
            safe_mag = mag.masked_fill(hidden, 0.0)
            magnitude_tokens = self.mag_projection(safe_mag.unsqueeze(-1))
            magnitude_tokens = torch.where(
                hidden[..., None],
                self.mask_embedding,
                magnitude_tokens,
            )
            pool_mask = point_mask & ~hidden

        tokens = (
            magnitude_tokens
            + self.band_embedding(band)
            + self.time_embedding(t)
        )
        if self.side_dim > 0 and side is not None:
            tokens = tokens + self.side_token(side)[:, None, :]
        x = self.token_norm(tokens)

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, point_mask, use_reentrant=False)
            else:
                x = block(x, point_mask)

        # Attention pooling with learned per-head queries over the visible points.
        batch_size, n_points, d_model = x.shape
        keys = self.pool_keys(x).reshape(
            batch_size, n_points, self.n_heads, self.head_dim
        )
        scores = torch.einsum("bnhd,hd->bhn", keys, self.pool_query)
        scores = scores / math.sqrt(self.head_dim)
        has_visible = pool_mask.any(dim=1, keepdim=True)
        safe_pool_mask = pool_mask | ~has_visible
        scores = scores.masked_fill(
            ~safe_pool_mask[:, None, :], float("-inf")
        )
        head_weights = scores.softmax(dim=-1)
        head_weights = head_weights * pool_mask[:, None, :].to(
            head_weights.dtype
        )
        pool_weights = head_weights.mean(dim=1)

        values = x.reshape(batch_size, n_points, self.n_heads, self.head_dim)
        pooled = torch.einsum("bhn,bnhd->bhd", head_weights, values)
        pooled = pooled.reshape(batch_size, d_model)
        pooled = self.pool_norm(self.pool_out(pooled))

        # Add light-curve metadata: log10 baseline and log10 number of points.
        n_real = point_mask.sum(dim=1).to(dtype=baseline.dtype)
        metadata = torch.stack(
            (
                baseline.clamp_min(1e-3).log10(),
                n_real.log10(),
            ),
            dim=-1,
        )
        z = pooled + self.metadata_mlp(metadata)
        return {"z": z, "tokens": x, "pool_weights": pool_weights}
