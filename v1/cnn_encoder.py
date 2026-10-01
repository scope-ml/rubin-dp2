# cnn_encoder.py
"""Baseline fold encoder: a 1D ResNet over phase bins with circular padding.

The folded light curve is binned in phase (n_bins bins). For each band the input has two
channels: the mean magnitude in each bin and log(1 + number of points in the bin). All
convolutions use circular padding, so phase 0 and phase 1 are neighbours. Global mean and max
pooling give one vector per fold. Used as a drop-in replacement for the transformer encoder
(FoldBranch(encoder="cnn")) to compare the two.
"""
import torch
from torch import nn


class _PreActivationBlock(nn.Module):
    """Pre-activation residual block (BatchNorm, GELU, circular conv, twice)."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 3, padding=1, padding_mode="circular"
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, 3, padding=1, padding_mode="circular"
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(self.activation(self.bn1(x)))
        x = self.conv2(self.dropout(self.activation(self.bn2(x))))
        return x + residual


class CircularCNNEncoder(nn.Module):
    """Encode one fold (binned in phase) into a vector with a circular-padding 1D ResNet."""

    def __init__(
        self,
        d_model: int = 64,
        n_bins: int = 64,
        n_bands: int = 6,
        widths: tuple[int, ...] = (32, 64, 64),
        blocks_per_stage: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_bins < 1 or n_bands < 1 or not widths or blocks_per_stage < 1:
            raise ValueError("n_bins, n_bands, and blocks_per_stage must be positive")
        if any(width < 1 for width in widths):
            raise ValueError("all widths must be positive")

        self.n_bins = n_bins
        self.n_bands = n_bands
        self.feature_width = widths[-1]

        self.stem = nn.Sequential(
            nn.Conv1d(
                2 * n_bands, widths[0], 5, padding=2, padding_mode="circular"
            ),
            nn.BatchNorm1d(widths[0]),
            nn.GELU(),
        )

        stages = []
        downsample = []
        for stage_index, width in enumerate(widths):
            stages.append(
                nn.Sequential(
                    *[
                        _PreActivationBlock(width, width, dropout)
                        for _ in range(blocks_per_stage)
                    ]
                )
            )
            if stage_index + 1 < len(widths):
                downsample.append(
                    nn.Conv1d(
                        width,
                        widths[stage_index + 1],
                        3,
                        stride=2,
                        padding=1,
                        padding_mode="circular",
                    )
                )
        self.stages = nn.ModuleList(stages)
        self.downsample = nn.ModuleList(downsample)
        self.projection = nn.Linear(2 * widths[-1], d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def bin_points(
        self,
        mag: torch.Tensor,
        band: torch.Tensor,
        phase: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Bin points by phase: per band, mean magnitude and log(1 + count) in each bin."""
        folds, _ = mag.shape
        valid_mag = torch.where(mask, mag, torch.zeros_like(mag))
        valid_band = torch.where(mask, band, torch.zeros_like(band))
        valid_phase = torch.where(mask, phase, torch.zeros_like(phase))
        bins = torch.floor(valid_phase * self.n_bins).long()
        bins = bins.clamp(0, self.n_bins - 1)

        fold_index = torch.arange(folds, device=mag.device)[:, None]
        flat_index = (
            (fold_index * self.n_bands + valid_band) * self.n_bins + bins
        ).reshape(-1)
        size = folds * self.n_bands * self.n_bins

        sums = mag.new_zeros(size)
        counts = mag.new_zeros(size)
        sums.scatter_add_(0, flat_index, valid_mag.reshape(-1))
        counts.scatter_add_(0, flat_index, mask.to(mag.dtype).reshape(-1))

        means = sums / counts.clamp_min(1)
        means = means.reshape(folds, self.n_bands, self.n_bins)
        occupancy = torch.log1p(counts).reshape(
            folds, self.n_bands, self.n_bins
        )
        return torch.cat((means, occupancy), dim=1)

    def forward(
        self,
        mag: torch.Tensor,
        band: torch.Tensor,
        phase: torch.Tensor,
        mask: torch.Tensor,
        return_features: bool = False,
    ):
        x = self.stem(self.bin_points(mag, band, phase, mask))
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.downsample):
                x = self.downsample[index](x)

        pooled = torch.cat((x.mean(dim=-1), x.amax(dim=-1)), dim=1)
        vector = self.output_norm(self.projection(pooled))
        if return_features:
            return vector, x
        return vector
