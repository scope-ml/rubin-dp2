# test_ce.py
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

import build_cache as cache_module
import pretrain
from cached_dataset import CachedLightCurveDataset
from pretrain import PretrainModel, conditional_entropy_ratio


def _ce_inputs(n_points=8192):
    generator = torch.Generator().manual_seed(12345)
    period = 2.0
    t = torch.sort(
        torch.rand(
            (2, n_points), generator=generator, dtype=torch.float64
        ) * (120.0 * period),
        dim=-1,
    ).values
    phase = torch.remainder(t[0] / period, 1.0)
    signal = torch.where(phase < 0.5, -2.0, 2.0).float()
    signal = signal + 0.04 * torch.randn(n_points, generator=generator)
    noise = torch.randn(n_points, generator=generator)
    return {
        "mag": torch.stack((signal, noise)),
        "t": t,
        "periods": torch.tensor(
            [[period, 1.37 * period, 0.61 * period]] * 2,
            dtype=torch.float32,
        ),
        "period_mask": torch.ones(2, 3, dtype=torch.bool),
        "point_mask": torch.ones(2, n_points, dtype=torch.bool),
        "hidden": torch.rand((2, n_points), generator=generator) < 0.3,
    }


def _model_batch(n_points=96):
    inputs = _ce_inputs(n_points)
    t = inputs["t"].float()
    baseline = t[:, -1] - t[:, 0]
    return {
        "t": t,
        "mag": inputs["mag"],
        "band": torch.arange(n_points).remainder(6)[None, :].expand(2, -1),
        "point_mask": inputs["point_mask"],
        "periods": inputs["periods"],
        "period_mask": inputs["period_mask"],
        "cycles": baseline[:, None] / inputs["periods"],
        "baseline": baseline,
        "side": torch.zeros(2, 11),
        "hidden": inputs["hidden"],
    }


def _model_options():
    return {
        "d_model": 16,
        "side_dim": 11,
        "normalize": True,
        "fvu_kind": "abs",
        "fvu_weight": 0.1,
        "rank_weight": 0.3,
        "rank_tau": 1.0,
        "fold_kwargs": {
            "n_heads": 2,
            "n_layers": 1,
            "n_cand_layers": 1,
            "dropout": 0.0,
        },
        "unfolded_kwargs": {
            "n_heads": 2,
            "n_layers": 1,
            "n_time_freqs": 4,
            "dropout": 0.0,
        },
    }


def test_ce_zero_is_bit_identical_and_skips_computation(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("CE must not be computed at zero weight")

    monkeypatch.setattr(pretrain, "conditional_entropy_ratio", unexpected)
    torch.manual_seed(42)
    default = PretrainModel(**_model_options())
    torch.manual_seed(42)
    explicit = PretrainModel(
        **_model_options(),
        ce_weight=0.0,
        ce_phase_bins=7,
        ce_mag_bins=3,
        ce_alpha=0.7,
    )
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key in default.state_dict():
        assert torch.equal(default.state_dict()[key], explicit.state_dict()[key])

    batch = _model_batch()
    with torch.no_grad():
        original = default(batch)
        modified = explicit(batch)
    assert original.keys() == modified.keys()
    for key in original:
        assert torch.equal(original[key], modified[key]), key
    assert original["ce_ratio_top_median"].item() == 0.0
    assert original["ce_ratio_best_median"].item() == 0.0


def test_ce_square_wave_and_noise():
    inputs = _ce_inputs()
    ratio = conditional_entropy_ratio(
        **inputs, phase_bins=10, mag_bins=5, alpha=0.5
    )
    assert ratio.shape == (2, 3)
    assert ratio.dtype == torch.float32
    assert torch.isfinite(ratio).all()
    assert not ratio.requires_grad

    assert ratio[0, 0].item() < 0.85
    assert torch.all(ratio[0, 1:] - ratio[0, 0] > 0.15)
    assert torch.all((ratio[1] - 1.0).abs() < 0.15)


def test_ce_forward_backward_changes_rank_but_not_fvu():
    torch.manual_seed(42)
    baseline = PretrainModel(**_model_options())
    torch.manual_seed(42)
    model = PretrainModel(**_model_options(), ce_weight=3.0)
    batch = _model_batch()
    before = {key: value.clone() for key, value in batch.items()}

    with torch.no_grad():
        old = baseline(batch)
    output = model(batch)
    assert output["loss"].device.type == "cpu"
    assert output["loss"].dtype == torch.float32
    assert torch.isfinite(output["loss"])
    assert torch.equal(output["loss_fvu"], old["loss_fvu"])
    assert torch.equal(output["loss_fold"], old["loss_fold"])
    assert torch.equal(output["loss_unf"], old["loss_unf"])
    assert abs(output["loss_rank"].item() - old["loss_rank"].item()) > 1e-6
    for key in ("ce_ratio_top_median", "ce_ratio_best_median"):
        assert torch.isfinite(output[key])
        assert not output[key].requires_grad
    assert output["loss_rank_kl"].item() >= -1e-5

    output["loss"].backward()
    for layer in (
        model.fold.side_point,
        model.fold.side_desc,
        model.unfolded.side_token,
    ):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    for key in batch:
        assert torch.equal(batch[key], before[key])


def test_ce_hidden_changes_scores_but_not_fitted_counts(monkeypatch):
    inputs = _ce_inputs(n_points=2048)
    before = {key: value.clone() for key, value in inputs.items()}
    records = []
    original_counts = pretrain._conditional_entropy_counts

    def capture(*args, **kwargs):
        counts, marginal = original_counts(*args, **kwargs)
        records.append((counts.clone(), marginal.clone()))
        return counts, marginal

    monkeypatch.setattr(pretrain, "_conditional_entropy_counts", capture)
    original = conditional_entropy_ratio(
        **inputs, phase_bins=10, mag_bins=5, alpha=0.5
    )
    changed_mag = inputs["mag"].clone()
    changed_mag[inputs["hidden"]] = -changed_mag[inputs["hidden"]]
    changed_inputs = dict(inputs, mag=changed_mag)
    modified = conditional_entropy_ratio(
        **changed_inputs, phase_bins=10, mag_bins=5, alpha=0.5
    )

    assert len(records) == 2
    counts_before, marginal_before = records[0]
    counts_after, marginal_after = records[1]
    assert torch.equal(counts_before, counts_after)
    assert torch.equal(marginal_before, marginal_after)
    probabilities_before = (
        (counts_before + 0.5)
        / (counts_before.sum(dim=-1, keepdim=True) + 0.5 * 5)
    )
    probabilities_after = (
        (counts_after + 0.5)
        / (counts_after.sum(dim=-1, keepdim=True) + 0.5 * 5)
    )
    assert torch.equal(probabilities_before, probabilities_after)

    n_visible = (
        inputs["point_mask"] & ~inputs["hidden"]
    ).sum(dim=-1).float()
    torch.testing.assert_close(
        counts_before.sum(dim=(-1, -2)),
        n_visible[:, None].expand(-1, 3),
        rtol=0,
        atol=0,
    )
    assert modified[0, 0].item() > original[0, 0].item() + 0.5
    for key in inputs:
        assert torch.equal(inputs[key], before[key])


def test_ce_cpu_bfloat16_autocast_keeps_float32_math():
    inputs = _ce_inputs(n_points=1024)
    expected = conditional_entropy_ratio(
        **inputs, phase_bins=10, mag_bins=5, alpha=0.5
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = conditional_entropy_ratio(
            **inputs, phase_bins=10, mag_bins=5, alpha=0.5
        )
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_ce_degenerate_objects_are_neutral():
    inputs = {
        "mag": torch.tensor(
            [[-2.0, 0.0, 1.0, 3.0]] * 3
        ),
        "t": torch.arange(4, dtype=torch.float64)[None, :].expand(3, -1),
        "periods": torch.tensor([[2.0, 0.0]] * 3),
        "period_mask": torch.tensor([[True, False]] * 3),
        "point_mask": torch.ones(3, 4, dtype=torch.bool),
        "hidden": torch.tensor(
            [[True, True, True, True],
             [False, False, False, False],
             [False, True, True, True]]
        ),
    }
    ratio = conditional_entropy_ratio(
        **inputs, phase_bins=10, mag_bins=5, alpha=0.5
    )
    assert torch.equal(ratio, torch.ones(3, 2))


def test_fold_quality_ce_changes_only_ranking_and_filters_eligible_rows():
    target = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0, 6.0]] * 3)
    hidden = torch.tensor(
        [[True] * 6, [False] * 6, [True] * 6]
    )
    period_mask = torch.tensor(
        [[True, True, False], [True, True, True], [False, False, False]]
    )
    pred = target[:, None, :].expand(-1, 3, -1).clone()
    scores = torch.tensor(
        [[1.0, 2.0, -torch.inf],
         [0.0, 0.0, 0.0],
         [-torch.inf, -torch.inf, -torch.inf]],
        requires_grad=True,
    )
    ce = torch.tensor(
        [[0.2, 1.3, -100.0],
         [torch.nan, torch.nan, torch.nan],
         [torch.nan, torch.nan, torch.nan]],
        requires_grad=True,
    )
    baseline = PretrainModel.fold_quality(
        pred, scores, target, hidden, period_mask, rank_tau=0.7
    )
    modified = PretrainModel.fold_quality(
        pred,
        scores,
        target,
        hidden,
        period_mask,
        rank_tau=0.7,
        ce_ratio=ce,
        ce_weight=1.0,
    )

    assert torch.equal(baseline[0], modified[0])
    q = torch.softmax(-torch.tensor([0.2, 1.3]) / 0.7, dim=-1)
    expected_rank = -(
        q * torch.log_softmax(torch.tensor([1.0, 2.0]), dim=-1)
    ).sum()
    torch.testing.assert_close(modified[1], expected_rank)
    assert not torch.isclose(baseline[1], modified[1])
    assert modified[2]["ce_ratio_top_median"].item() == pytest.approx(1.3)
    assert modified[2]["ce_ratio_best_median"].item() == pytest.approx(0.2)
    for value in modified[2].values():
        assert torch.isfinite(value)
        assert not value.requires_grad

    modified[1].backward()
    assert ce.grad is None
    assert torch.isfinite(scores.grad).all()


@pytest.mark.parametrize("case", ["no_hidden", "no_valid"])
def test_fold_quality_ce_no_eligible_diagnostics(case):
    hidden = torch.ones(1, 4, dtype=torch.bool)
    period_mask = torch.ones(1, 2, dtype=torch.bool)
    if case == "no_hidden":
        hidden.zero_()
    else:
        period_mask.zero_()
    loss_fvu, loss_rank, diagnostics = PretrainModel.fold_quality(
        torch.zeros(1, 2, 4),
        torch.zeros(1, 2).masked_fill(~period_mask, -torch.inf),
        torch.zeros(1, 4),
        hidden,
        period_mask,
        ce_ratio=torch.ones(1, 2),
        ce_weight=1.0,
    )
    assert loss_fvu.item() == 0.0
    assert loss_rank.item() == 0.0
    assert diagnostics["ce_ratio_top_median"].item() == 0.0
    assert diagnostics["ce_ratio_best_median"].item() == 0.0
    for value in diagnostics.values():
        assert torch.isfinite(value)
        assert not value.requires_grad


@pytest.mark.parametrize(
    "options",
    [
        {"ce_weight": -0.1},
        {"ce_phase_bins": 1},
        {"ce_mag_bins": 1},
        {"ce_alpha": 0.0},
        {"ce_alpha": -0.5},
    ],
)
def test_ce_options_validation(options):
    with pytest.raises(ValueError, match="ce_"):
        PretrainModel(**options)


def _make_parquets(tmp_path):
    rows = []
    candidates = []
    for object_id in (101, 202, 303):
        for point in range(24):
            rows.append(
                {
                    "diaObjectId": object_id,
                    "mjd": 60000.0 + point,
                    "band": "r" if point % 2 else "g",
                    "mag": 19.0 + 0.1 * math.sin(
                        2.0 * math.pi * point / (1.5 + object_id * 0.001)
                    ),
                }
            )
        candidates.append(
            {
                "_id": object_id,
                "period_1_LS": 1.5 + object_id * 0.001,
                "period_1_CE": 2.7 + object_id * 0.001,
            }
        )
    lc = tmp_path / "part.parquet"
    cands = tmp_path / "cands.parquet"
    pd.DataFrame(rows).to_parquet(lc, index=False)
    pd.DataFrame(candidates).to_parquet(cands, index=False)
    return lc, cands


def test_build_cache_ids_cli(tmp_path):
    lc, cands = _make_parquets(tmp_path)
    ids = tmp_path / "ids.csv"
    pd.DataFrame({"oid": [101, 303]}).to_csv(ids, index=False)
    out = tmp_path / "selected"

    cache_module.main(
        ["--lc", str(lc),
         "--cands", str(cands),
         "--out", str(out),
         "--ids", str(ids),
         "--workers", "1",
         "--shard-objects", "1",
         "--spike-floor", "none",
         "--min-cadence-minutes", "none",
         "--mag-transform", "none",
         "--scale-kind", "none"]
    )
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    cached = CachedLightCurveDataset(out, train=False)
    assert len(cached) == 2
    np.testing.assert_array_equal(cached.ids, [101, 303])
    assert meta["counts"]["objects"] == 2
    assert meta["counts"]["points"] == 48
    assert meta["counts"]["skipped_objects"] == 0
    assert meta["inputs"]["ids"]["path"] == str(ids.resolve())
    assert meta["inputs"]["ids"]["count"] == 2

    default_out = tmp_path / "all"
    default_meta = cache_module.build_cache(
        str(lc),
        cands,
        default_out,
        spike_floor=None,
        min_cadence_minutes=None,
        mag_transform=None,
        scale_kind="none",
    )
    assert default_meta["counts"]["objects"] == 3
    assert "ids" not in default_meta["inputs"]
    np.testing.assert_array_equal(
        CachedLightCurveDataset(default_out, train=False).ids,
        [101, 202, 303],
    )


def test_worker_ids_filter_precedes_skip_counting(tmp_path, monkeypatch):
    lc, _ = _make_parquets(tmp_path)
    monkeypatch.setattr(
        cache_module, "_CANDIDATES", {101: [2.0], 303: [2.0]}
    )
    monkeypatch.setattr(
        cache_module,
        "_BUILD_OPTIONS",
        {
            "min_points": 20,
            "spike_floor": None,
            "min_cadence_minutes": None,
            "mag_transform": None,
            "scale_kind": "none",
        },
    )
    monkeypatch.setattr(cache_module, "_IDS", {101, 303})
    objects, counts = cache_module._process_part(lc)
    assert [int(obj["id"]) for obj in objects] == [101, 303]
    assert counts["objects"] == 2
    assert counts.get("no_candidates", 0) == 0
