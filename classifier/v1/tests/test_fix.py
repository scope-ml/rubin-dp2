# test_fix.py
import math

import numpy as np
import pandas as pd
import pytest
import torch

from dataset import LightCurveDataset, build_object
from pretrain import PretrainModel
from train_pretrain import parse_args


def light_curve_rows():
    return pd.DataFrame(
        {
            "diaObjectId": [42] * 5,
            "mjd": [10.0, 11.0, 12.0, 13.0, 14.0],
            "band": ["r"] * 5,
            "mag": [8.0, 9.0, 10.0, 11.0, 12.0],
        }
    )


def test_asinh_is_applied_after_robust_scaling():
    rows = light_curve_rows()
    candidate = {"period_1_LS": 2.0}
    old = build_object(
        rows,
        candidate,
        min_points=5,
        spike_floor=None,
        min_cadence_minutes=None,
        mag_transform=None,
    )
    transformed = build_object(
        rows,
        candidate,
        min_points=5,
        spike_floor=None,
        min_cadence_minutes=None,
        mag_transform="asinh",
    )
    default = build_object(
        rows,
        candidate,
        min_points=5,
        spike_floor=None,
        min_cadence_minutes=None,
    )

    scale = 1.4826
    expected_old = (
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / scale
    ).astype(np.float32)
    expected_new = np.arcsinh(
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / scale
    ).astype(np.float32)

    np.testing.assert_array_equal(old["mag"], expected_old)
    np.testing.assert_array_equal(transformed["mag"], expected_new)
    np.testing.assert_array_equal(default["mag"], expected_new)
    assert transformed["mag"][0] == -transformed["mag"][-1]


def test_dataset_passes_transform_through(tmp_path):
    pytest.importorskip("pyarrow")
    rows = light_curve_rows()
    candidates = pd.DataFrame(
        {"_id": [42], "period_1_LS": [2.0]}
    )
    lc_path = tmp_path / "lc.parquet"
    cands_path = tmp_path / "cands.parquet"
    rows.to_parquet(lc_path)
    candidates.to_parquet(cands_path)

    common = dict(
        min_points=5,
        train=False,
        spike_floor=None,
        min_cadence_minutes=None,
    )
    old = LightCurveDataset(
        lc_path, cands_path, mag_transform=None, **common
    )
    new = LightCurveDataset(
        lc_path, cands_path, mag_transform="asinh", **common
    )
    np.testing.assert_array_equal(
        new[0]["mag"], np.arcsinh(old[0]["mag"]).astype(np.float32)
    )


def test_laplace_mixture_matches_hand_calculation():
    target = torch.tensor([[0.0, 1.0]])
    hidden = torch.tensor([[True, True]])
    periods = torch.tensor([[True, True]])
    predictions = torch.tensor(
        [[[0.0, 1.0], [1.0, 2.0]]]
    )
    log_b = torch.tensor(math.log(0.5))
    good_scores = torch.tensor([[2.0, -2.0]])
    bad_scores = torch.tensor([[-2.0, 2.0]])

    good_loss, diagnostics = PretrainModel.fold_mixture(
        predictions,
        good_scores,
        target,
        hidden,
        periods,
        log_b,
        likelihood="laplace",
    )
    bad_loss, _ = PretrainModel.fold_mixture(
        predictions,
        bad_scores,
        target,
        hidden,
        periods,
        log_b,
        likelihood="laplace",
    )

    log_weights = torch.log_softmax(good_scores, dim=-1)[0]
    ll_good = -2 * math.log(1.0)
    ll_bad = -2.0 / 0.5 - 2 * math.log(1.0)
    expected = -torch.logsumexp(
        log_weights
        + torch.tensor([ll_good, ll_bad]),
        dim=0,
    ) / 2
    torch.testing.assert_close(good_loss, expected)
    assert good_loss < bad_loss
    assert diagnostics["top_is_best"].item() == 1.0
    assert diagnostics["mse_best_fold"].item() == 0.0
    assert diagnostics["mse_top_fold"].item() == 0.0


def test_gaussian_mode_reproduces_previous_fixed_batch_loss():
    target = torch.tensor([[0.0, 1.0, 5.0]])
    hidden = torch.tensor([[True, True, False]])
    periods = torch.tensor([[True, True, False]])
    predictions = torch.tensor(
        [[[0.25, 0.75, 99.0], [1.0, 2.0, -99.0], [0.0, 0.0, 0.0]]]
    )
    scores = torch.tensor([[1.5, -0.5, -torch.inf]])
    log_sigma = torch.tensor(math.log(0.5))
    loss, diagnostics = PretrainModel.fold_mixture(
        predictions,
        scores,
        target,
        hidden,
        periods,
        log_sigma,
        likelihood="gaussian",
    )

    variance = 0.5**2
    log_normal_constant = math.log(0.5) + 0.5 * math.log(
        2.0 * math.pi
    )
    ll0 = -0.5 * (0.25**2 + 0.25**2) / variance - 2 * log_normal_constant
    ll1 = -0.5 * (1.0**2 + 1.0**2) / variance - 2 * log_normal_constant
    expected = -torch.logsumexp(
        torch.log_softmax(scores[:, :2], dim=-1)[0]
        + torch.tensor([ll0, ll1]),
        dim=0,
    ) / 2
    torch.testing.assert_close(loss, expected)
    assert diagnostics["mse_best_fold"].item() == pytest.approx(
        0.25**2
    )

    model = PretrainModel(
        likelihood="gaussian",
        d_model=16,
        fold_kwargs={"n_layers": 0, "n_cand_layers": 0},
        unfolded_kwargs={"n_layers": 0},
    )
    assert "log_sigma" in model.state_dict()
    assert "log_b" not in model.state_dict()


def test_cli_defaults_and_explicit_compatibility_options():
    base = ["--lc", "lc.parquet", "--cands", "cands.parquet", "--out", "run"]
    defaults = parse_args(base)
    assert defaults.mag_transform == "asinh"
    assert defaults.likelihood == "laplace"

    old = parse_args(
        base + ["--mag-transform", "none", "--likelihood", "gaussian"]
    )
    assert old.mag_transform == "none"
    assert old.likelihood == "gaussian"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
def test_laplace_loss_is_finite_under_bf16_autocast():
    device = "cuda"
    torch.manual_seed(9)
    model = PretrainModel(
        likelihood="laplace",
        d_model=16,
        fold_kwargs={
            "n_heads": 2,
            "n_layers": 1,
            "n_cand_layers": 1,
            "fold_chunk": 2,
        },
        unfolded_kwargs={"n_heads": 2, "n_layers": 1},
    ).to(device)
    batch = {
        "t": torch.arange(12, device=device).float()[None, :],
        "band": torch.zeros(1, 12, dtype=torch.long, device=device),
        "mag": torch.tensor(
            [[0.0, 0.2, -0.1, 2.0, -3.0, 0.3,
              0.4, -0.2, 0.1, 0.0, 1.5, -1.0]],
            device=device,
        ),
        "point_mask": torch.ones(
            1, 12, dtype=torch.bool, device=device
        ),
        "hidden": torch.tensor(
            [[False, False, True, True, False, False,
              True, False, False, False, False, False]],
            device=device,
        ),
        "periods": torch.tensor([[1.5, 2.7]], device=device),
        "cycles": torch.tensor([[7.3, 4.1]], device=device),
        "period_mask": torch.ones(
            1, 2, dtype=torch.bool, device=device
        ),
        "baseline": torch.tensor([11.0], device=device),
        "id": torch.tensor([1], device=device),
    }

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
    for value in output.values():
        assert torch.isfinite(value).all()
    output["loss"].backward()
    assert torch.isfinite(model.log_b.grad).all()
