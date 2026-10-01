# test_unfolded_branch.py
import math

import pytest
import torch

from unfolded_branch import UnfoldedBranch


def make_batch(batch_size=2, n_points=32, device="cpu"):
    generator = torch.Generator().manual_seed(123)
    return {
        "t": torch.rand(batch_size, n_points, generator=generator)
        .to(device)
        .mul(1200.0),
        "band": torch.randint(
            0, 6, (batch_size, n_points), generator=generator
        ).to(device),
        "mag": torch.randn(
            batch_size, n_points, generator=generator
        ).to(device),
        "point_mask": torch.ones(
            batch_size, n_points, dtype=torch.bool, device=device
        ),
        "baseline": torch.full(
            (batch_size,), 1200.0, dtype=torch.float32, device=device
        ),
    }


def test_point_permutation_invariance():
    model = UnfoldedBranch().eval()
    batch = make_batch()
    permutation = torch.randperm(batch["t"].shape[1])

    shuffled = dict(batch)
    for key in ("t", "band", "mag", "point_mask"):
        shuffled[key] = batch[key][:, permutation]

    with torch.no_grad():
        original = model(batch)
        permuted = model(shuffled)

    torch.testing.assert_close(original["z"], permuted["z"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        original["tokens"][:, permutation],
        permuted["tokens"],
        atol=1e-4,
        rtol=1e-4,
    )


def test_padding_invariance():
    model = UnfoldedBranch().eval()
    batch = make_batch(n_points=25)
    padded = dict(batch)
    extra = 9
    padded["t"] = torch.cat(
        (batch["t"], torch.full((2, extra), 9000.0)), dim=1
    )
    padded["band"] = torch.cat(
        (batch["band"], torch.full((2, extra), 5, dtype=torch.long)), dim=1
    )
    padded["mag"] = torch.cat(
        (batch["mag"], torch.full((2, extra), 100.0)), dim=1
    )
    padded["point_mask"] = torch.cat(
        (batch["point_mask"], torch.zeros((2, extra), dtype=torch.bool)),
        dim=1,
    )

    with torch.no_grad():
        original = model(batch)
        with_padding = model(padded)

    torch.testing.assert_close(
        original["z"], with_padding["z"], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        original["tokens"],
        with_padding["tokens"][:, :25],
        atol=1e-4,
        rtol=1e-4,
    )


def test_time_features_use_float64_angle():
    model = UnfoldedBranch(t_min=0.005).eval()
    features = model.time_embedding.features(torch.tensor([[250.0]]))
    angle = torch.tensor(250.0, dtype=torch.float64) * (
        2.0 * math.pi / torch.tensor(0.005, dtype=torch.float64)
    )
    n_freqs = model.time_embedding.angular_frequencies.numel()

    torch.testing.assert_close(
        features[0, 0, 0],
        angle.sin().float(),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        features[0, 0, n_freqs],
        angle.cos().float(),
        atol=1e-5,
        rtol=0,
    )


def test_sparse_and_long_objects():
    model = UnfoldedBranch().eval()
    sparse = make_batch(batch_size=1, n_points=20)
    sparse["band"].zero_()
    long = make_batch(batch_size=1, n_points=1800)
    long["band"][0, :6] = torch.arange(6)

    with torch.no_grad():
        sparse_output = model(sparse)
        long_output = model(long)

    assert sparse_output["z"].shape == (1, 64)
    assert sparse_output["tokens"].shape == (1, 20, 64)
    assert long_output["z"].shape == (1, 64)
    assert long_output["tokens"].shape == (1, 1800, 64)


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_forward_backward_finite_and_all_parameters_receive_gradients(
    device, grad_checkpoint
):
    model = UnfoldedBranch(
        n_layers=2, grad_checkpoint=grad_checkpoint
    ).to(device)
    model.train()
    batch = make_batch(n_points=24, device=device)
    output = model(batch)
    loss = output["z"].square().mean() + output["tokens"].square().mean()
    loss.backward()

    assert torch.isfinite(output["z"]).all()
    assert torch.isfinite(output["tokens"]).all()
    for name, parameter in model.named_parameters():
        if "mask_embedding" in name:  # used only when the batch has hidden points
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_pool_weights_respect_mask():
    model = UnfoldedBranch().eval()
    batch = make_batch()
    batch["point_mask"][:, -7:] = False

    with torch.no_grad():
        weights = model(batch)["pool_weights"]

    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones(weights.shape[0]),
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.equal(
        weights[~batch["point_mask"]],
        torch.zeros_like(weights[~batch["point_mask"]]),
    )
