# test_cnn_encoder.py
import pytest
import torch

from cnn_encoder import CircularCNNEncoder


def _sparse_folds():
    generator = torch.Generator().manual_seed(17)
    folds, points, n_bins = 4, 160, 64
    mag = torch.randn(folds, points, generator=generator)
    band = torch.randint(0, 6, (folds, points), generator=generator)
    bins = torch.randint(0, n_bins, (folds, points), generator=generator)
    phase = (bins.float() + 0.25) / n_bins
    mask = torch.rand(folds, points, generator=generator) > 0.15
    return mag, band, phase, mask


def _smooth_folds():
    n_bins, n_bands, folds = 64, 6, 3
    bin_index = torch.arange(n_bins)
    phase = ((bin_index.float() + 0.25) / n_bins).repeat(folds, n_bands)
    band = torch.arange(n_bands).repeat_interleave(n_bins).repeat(folds, 1)
    angle = 2 * torch.pi * (bin_index.float() + 0.25) / n_bins
    mag = torch.stack(
        [
            torch.cat(
                [
                    torch.sin(angle + 0.3 * fold + 0.2 * b)
                    + 0.2 * torch.cos(2 * angle - 0.1 * b)
                    for b in range(n_bands)
                ]
            )
            for fold in range(folds)
        ]
    )
    return mag, band, phase, torch.ones_like(band, dtype=torch.bool)


@pytest.mark.parametrize("shift", [1, 7, 32])
def test_whole_bin_phase_shift(shift):
    torch.manual_seed(23)
    model = CircularCNNEncoder(widths=(16,), blocks_per_stage=1).eval()
    mag, band, phase, mask = _sparse_folds()
    shifted = (phase + shift / model.n_bins) % 1
    with torch.no_grad():
        original = model(mag, band, phase, mask)
        result = model(mag, band, shifted, mask)
    torch.testing.assert_close(result, original, atol=1e-5, rtol=1e-5)

    default_model = CircularCNNEncoder().eval()
    mag, band, phase, mask = _smooth_folds()
    shifted = (phase + shift / default_model.n_bins) % 1
    with torch.no_grad():
        original = default_model(mag, band, phase, mask)
        result = default_model(mag, band, shifted, mask)
    relative_change = (result - original).norm(dim=1) / original.norm(dim=1)
    assert torch.all(relative_change < 0.05)


def test_point_permutation():
    torch.manual_seed(29)
    model = CircularCNNEncoder().eval()
    mag, band, phase, mask = _sparse_folds()
    permutation = torch.randperm(mag.shape[1])
    with torch.no_grad():
        original = model(mag, band, phase, mask)
        permuted = model(
            mag[:, permutation],
            band[:, permutation],
            phase[:, permutation],
            mask[:, permutation],
        )
    torch.testing.assert_close(permuted, original, atol=1e-5, rtol=1e-5)


def test_appending_masked_points():
    model = CircularCNNEncoder().eval()
    mag, band, phase, mask = _sparse_folds()
    folds = mag.shape[0]
    with torch.no_grad():
        original = model(mag, band, phase, mask)
        appended = model(
            torch.cat((mag, torch.full((folds, 5), float("nan"))), dim=1),
            torch.cat((band, torch.zeros(folds, 5, dtype=torch.long)), dim=1),
            torch.cat((phase, torch.full((folds, 5), float("nan"))), dim=1),
            torch.cat((mask, torch.zeros(folds, 5, dtype=torch.bool)), dim=1),
        )
    torch.testing.assert_close(appended, original, atol=1e-5, rtol=1e-5)


def test_hand_computed_binning():
    model = CircularCNNEncoder(
        n_bins=4, n_bands=2, widths=(8,), blocks_per_stage=1
    )
    binned = model.bin_points(
        torch.tensor([[2.0, 4.0, 10.0]]),
        torch.tensor([[0, 0, 1]]),
        torch.tensor([[0.1, 0.2, 0.9]]),
        torch.tensor([[True, True, True]]),
    )
    expected = torch.zeros(1, 4, 4)
    expected[0, 0, 0] = 3.0
    expected[0, 1, 3] = 10.0
    expected[0, 2, 0] = torch.log1p(torch.tensor(2.0))
    expected[0, 3, 3] = torch.log1p(torch.tensor(1.0))
    torch.testing.assert_close(binned, expected, atol=1e-5, rtol=1e-5)


def test_missing_band_and_empty_fold():
    model = CircularCNNEncoder().eval()
    mag = torch.tensor([[1.0, -1.0, 2.0], [0.0, 0.0, 0.0]])
    band = torch.tensor([[2, 2, 2], [0, 0, 0]])
    phase = torch.tensor([[0.1, 0.4, 0.9], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True], [False, False, False]])
    binned = model.bin_points(mag, band, phase, mask)
    assert torch.count_nonzero(binned[0, :2]) == 0
    assert torch.count_nonzero(binned[0, 3:6]) == 0
    assert torch.count_nonzero(binned[1]) == 0
    with torch.no_grad():
        output = model(mag, band, phase, mask)
    assert output.shape == (2, 64)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_forward_backward(device):
    torch.manual_seed(31)
    model = CircularCNNEncoder().to(device)
    folds, points = 10, 40
    mag = torch.randn(folds, points, device=device, requires_grad=True)
    band = torch.randint(0, 6, (folds, points), device=device)
    phase = torch.rand(folds, points, device=device)
    mask = torch.rand(folds, points, device=device) > 0.2

    output = model(mag, band, phase, mask)
    assert output.shape == (folds, 64)
    assert torch.isfinite(output).all()
    output.square().mean().backward()

    assert mag.grad is not None and torch.isfinite(mag.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
