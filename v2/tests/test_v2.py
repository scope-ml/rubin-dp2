# test_v2.py
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

import build_cache
from cached_dataset import CachedLightCurveDataset
from dataset import LightCurveDataset, build_object, collate, normalize_batch
from fold_branch import FoldBranch
from pretrain import PretrainModel
from unfolded_branch import UnfoldedBranch


def _rows(object_id=1):
    return pd.DataFrame(
        {
            "diaObjectId": np.full(9, object_id, dtype=np.int64),
            "mjd": 60000.0 + np.arange(9) * 0.7,
            "band": np.repeat(["u", "g", "r"], 3),
            "mag": [8.0, 10.0, 12.0, 7.0, 8.0, 9.0, 2.0, 5.0, 8.0],
        }
    )


def _object(object_id=1, **kwargs):
    options = {
        "min_points": 1,
        "spike_floor": None,
        "min_cadence_minutes": None,
        "mag_transform": None,
    }
    options.update(kwargs)
    return build_object(
        _rows(object_id), [1.0, 2.3, 4.0], **options
    )


def _batch(hidden=False):
    batch = collate([_object(1), _object(2)])
    if hidden:
        batch["hidden"] = torch.zeros_like(batch["point_mask"])
        batch["hidden"][:, ::2] = True
    return batch


def _write_meta(path, objects, shards):
    meta = {
        "format_version": 1,
        "counts": {
            "shards": shards,
            "objects": len(objects),
            "points": sum(len(obj["t"]) for obj in objects),
            "periods": sum(len(obj["periods"]) for obj in objects),
        },
    }
    (path / "meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def test_build_object_std_and_side():
    obj = _object(scale_kind="std")
    centered = np.array(
        [-2.0, 0.0, 2.0, -1.0, 0.0, 1.0, -3.0, 0.0, 3.0]
    )
    scale = np.std(centered, ddof=0)

    assert obj is not None
    assert np.std(obj["mag"], ddof=0) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(obj["mag"], centered / scale, rtol=1e-6)
    assert obj["side"].shape == (11,)
    assert obj["side"].dtype == np.float32
    np.testing.assert_allclose(
        obj["side"],
        [2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
         np.log10(scale)],
        rtol=1e-6,
        atol=1e-7,
    )

    transformed = _object(scale_kind="std", mag_transform="asinh")
    np.testing.assert_allclose(
        transformed["mag"], np.arcsinh(centered / scale), rtol=1e-6
    )
    np.testing.assert_array_equal(transformed["side"], obj["side"])


def test_side_uses_surviving_points():
    rows = pd.DataFrame(
        {
            "diaObjectId": np.ones(7, dtype=np.int64),
            "mjd": [0.0, 0.001, 1.0, 0.0, 1.0, 0.0, 1.0],
            "band": ["u", "u", "u", "g", "g", "r", "r"],
            "mag": [10.0, 99.0, 14.0, 8.0, 12.0, 5.0, 9.0],
        }
    )
    obj = build_object(
        rows,
        [1.0],
        min_points=1,
        spike_floor=None,
        min_cadence_minutes=5.0,
        mag_transform=None,
        scale_kind="std",
    )
    assert len(obj["t"]) == 6
    np.testing.assert_array_equal(
        obj["side"][:10],
        [2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    )
    assert obj["side"][10] == pytest.approx(np.log10(2.0), abs=1e-7)


@pytest.mark.parametrize("mag_transform", [None, "asinh"])
def test_mad_matches_default_and_original_formula(mag_transform):
    default = _object(mag_transform=mag_transform)
    explicit = _object(scale_kind="mad", mag_transform=mag_transform)
    assert default.keys() == explicit.keys()
    for key in default:
        np.testing.assert_array_equal(default[key], explicit[key])

    centered = np.array(
        [-2.0, 0.0, 2.0, -1.0, 0.0, 1.0, -3.0, 0.0, 3.0]
    )
    scale = max(1.4826 * np.median(np.abs(centered)), 1e-3)
    expected = centered / scale
    if mag_transform == "asinh":
        expected = np.arcsinh(expected)
    np.testing.assert_array_equal(default["mag"], expected.astype(np.float32))
    assert default["side"][10] == pytest.approx(np.log10(scale), abs=1e-7)


@pytest.mark.parametrize("scale_kind", ["mad", "std"])
def test_scale_floor(scale_kind):
    rows = _rows()
    rows["mag"] = np.repeat([10.0, 8.0, 5.0], 3)
    obj = build_object(
        rows,
        [1.0],
        min_points=1,
        spike_floor=None,
        min_cadence_minutes=None,
        mag_transform=None,
        scale_kind=scale_kind,
    )
    np.testing.assert_array_equal(obj["mag"], np.zeros(9, dtype=np.float32))
    assert obj["side"][10] == pytest.approx(-3.0)


def test_scale_kind_validation():
    with pytest.raises(ValueError, match="scale_kind"):
        _object(scale_kind="invalid")
    with pytest.raises(ValueError, match="scale_kind"):
        LightCurveDataset([], "unused", scale_kind="invalid")
    with pytest.raises(ValueError, match="scale_kind"):
        build_cache.build_cache(
            "unused", "unused", "unused", scale_kind="invalid"
        )


def test_light_curve_dataset_threads_scale_kind(monkeypatch):
    rows = _rows()
    candidates = pd.DataFrame(
        {
            "_id": [1],
            "period_1_LS": [1.0],
            "period_2_LS": [2.3],
            "period_3_LS": [4.0],
        }
    )

    def read_parquet(path, columns=None):
        frame = candidates if str(path) == "cands.parquet" else rows
        return frame.copy() if columns is None else frame[columns].copy()

    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    dataset = LightCurveDataset(
        ["lc.parquet"],
        "cands.parquet",
        min_points=1,
        train=False,
        spike_floor=None,
        min_cadence_minutes=None,
        mag_transform=None,
        scale_kind="std",
    )
    expected = _object(scale_kind="std")
    assert dataset.scale_kind == "std"
    assert len(dataset) == 1
    np.testing.assert_array_equal(dataset[0]["mag"], expected["mag"])
    np.testing.assert_array_equal(dataset[0]["side"], expected["side"])


def test_collate_missing_side_is_zero():
    current = _object(1)
    legacy = _object(2)
    del legacy["side"]
    batch = collate([current, legacy])
    assert batch["side"].shape == (2, 11)
    assert batch["side"].dtype == torch.float32
    torch.testing.assert_close(
        batch["side"][0], torch.from_numpy(current["side"])
    )
    torch.testing.assert_close(batch["side"][1], torch.zeros(11))


@pytest.mark.parametrize("with_side", [True, False])
def test_cache_round_trip(tmp_path, with_side):
    objects = [_object(1), _object(2)]
    objects[1]["side"] = objects[1]["side"] + np.float32(0.25)
    if not with_side:
        for obj in objects:
            del obj["side"]

    for index, obj in enumerate(objects):
        path = tmp_path / f"shard_{index:04d}.npz"
        build_cache._write_shard(path, [obj])
        with np.load(path, allow_pickle=False) as shard:
            assert ("side" in shard) == with_side
            if with_side:
                assert shard["side"].shape == (1, 11)
                assert shard["side"].dtype == np.float32
    _write_meta(tmp_path, objects, shards=2)

    dataset = CachedLightCurveDataset(tmp_path, train=False)
    assert len(dataset) == 2
    if with_side:
        assert dataset.side.shape == (2, 11)
    for index, obj in enumerate(objects):
        item = dataset[index]
        assert item.keys() == obj.keys()
        for key in obj:
            np.testing.assert_array_equal(item[key], obj[key])
    if not with_side:
        torch.testing.assert_close(
            collate([dataset[0], dataset[1]])["side"],
            torch.zeros(2, 11),
        )


def test_cache_mixed_shards(tmp_path):
    legacy = _object(1)
    del legacy["side"]
    current = _object(2)
    build_cache._write_shard(tmp_path / "shard_0000.npz", [legacy])
    build_cache._write_shard(tmp_path / "shard_0001.npz", [current])
    _write_meta(tmp_path, [legacy, current], shards=2)

    dataset = CachedLightCurveDataset(tmp_path, train=False)
    assert "side" not in dataset[0]
    np.testing.assert_array_equal(dataset[1]["side"], current["side"])
    batch = collate([dataset[0], dataset[1]])
    torch.testing.assert_close(batch["side"][0], torch.zeros(11))
    torch.testing.assert_close(
        batch["side"][1], torch.from_numpy(current["side"])
    )


def test_write_shard_omits_side_if_any_object_lacks_it(tmp_path):
    current = _object(1)
    legacy = _object(2)
    del legacy["side"]
    path = tmp_path / "shard_0000.npz"
    build_cache._write_shard(path, [current, legacy])
    with np.load(path, allow_pickle=False) as shard:
        assert "side" not in shard


def test_build_cache_std_metadata_and_cli(tmp_path, monkeypatch):
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    _rows().to_parquet(lc_dir / "part.parquet", index=False)
    cands_path = tmp_path / "cands.parquet"
    pd.DataFrame(
        {"_id": [1], "period_1_LS": [1.0], "period_2_LS": [2.3]}
    ).to_parquet(cands_path, index=False)
    out = tmp_path / "cache"

    meta = build_cache.build_cache(
        lc_dir,
        cands_path,
        out,
        min_points=1,
        spike_floor=None,
        min_cadence_minutes=None,
        mag_transform=None,
        scale_kind="std",
    )
    assert meta["parameters"]["scale_kind"] == "std"
    assert meta["parameters"]["mag_transform"] is None
    assert json.loads((out / "meta.json").read_text(encoding="utf-8")) == meta
    item = CachedLightCurveDataset(out, train=False)[0]
    assert np.std(item["mag"], ddof=0) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_array_equal(item["side"], _object(scale_kind="std")["side"])

    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(build_cache, "build_cache", capture)
    arguments = ["--lc", "lc", "--cands", "cands", "--out", "out"]
    build_cache.main(arguments)
    assert calls[-1]["scale_kind"] == "mad"
    assert calls[-1]["mag_transform"] == "asinh"
    build_cache.main(
        arguments + ["--scale-kind", "std", "--mag-transform", "none"]
    )
    assert calls[-1]["scale_kind"] == "std"
    assert calls[-1]["mag_transform"] is None


@pytest.mark.parametrize("encoder", ["rope", "cnn"])
def test_fold_branch_side_changes_output(encoder):
    torch.manual_seed(7)
    model = FoldBranch(
        d_model=16,
        n_heads=2,
        n_layers=1,
        n_cand_layers=1,
        dropout=0.0,
        encoder=encoder,
        side_dim=11,
    ).eval()
    batch = _batch()
    changed = dict(batch, side=batch["side"] + 1.0)
    with torch.no_grad():
        original = model(batch)
        modified = model(changed)
    assert original.keys() == {"z", "cand_scores", "cand_vecs"}
    assert modified.keys() == original.keys()
    assert not torch.allclose(original["z"], modified["z"])
    assert not torch.allclose(original["cand_vecs"], modified["cand_vecs"])
    assert torch.count_nonzero(model.side_desc.weight).item() > 0
    assert hasattr(model, "side_point") == (encoder == "rope")
    if encoder == "rope":
        assert torch.count_nonzero(model.side_point.weight).item() > 0


def test_unfolded_branch_side_changes_output():
    torch.manual_seed(8)
    model = UnfoldedBranch(
        d_model=16,
        n_heads=2,
        n_layers=1,
        n_time_freqs=4,
        dropout=0.0,
        side_dim=11,
    ).eval()
    batch = _batch()
    changed = dict(batch, side=batch["side"] + 1.0)
    with torch.no_grad():
        original = model(batch)
        modified = model(changed)
    assert original.keys() == {"z", "tokens", "pool_weights"}
    assert modified.keys() == original.keys()
    assert not torch.allclose(original["z"], modified["z"])
    assert not torch.allclose(original["tokens"], modified["tokens"])
    assert torch.count_nonzero(model.side_token.weight).item() > 0


@pytest.mark.parametrize("branch", ["fold", "unfolded"])
def test_side_dim_zero_preserves_outputs_and_state_keys(branch):
    if branch == "fold":
        factory = FoldBranch
        options = {
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "n_cand_layers": 1,
            "dropout": 0.0,
        }
        expected_keys = {"z", "cand_scores", "cand_vecs"}
    else:
        factory = UnfoldedBranch
        options = {
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "n_time_freqs": 4,
            "dropout": 0.0,
        }
        expected_keys = {"z", "tokens", "pool_weights"}

    torch.manual_seed(9)
    default = factory(**options).eval()
    torch.manual_seed(9)
    explicit = factory(**options, side_dim=0).eval()
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert not any(key.startswith("side_") for key in explicit.state_dict())

    batch = _batch()
    legacy = {key: value for key, value in batch.items() if key != "side"}
    changed = dict(batch, side=batch["side"] + 100.0)
    with torch.no_grad():
        original = default(legacy)
        with_side = explicit(batch)
        modified = explicit(changed)
    assert original.keys() == with_side.keys() == modified.keys() == expected_keys
    for key in original:
        torch.testing.assert_close(original[key], with_side[key], rtol=0, atol=0)
        torch.testing.assert_close(original[key], modified[key], rtol=0, atol=0)


@pytest.mark.parametrize("likelihood", ["laplace", "gaussian"])
def test_pretrain_abs_side_cpu_backward(likelihood):
    torch.manual_seed(10)
    model = PretrainModel(
        d_model=16,
        side_dim=11,
        fvu_kind="abs",
        likelihood=likelihood,
        fvu_weight=0.1,
        rank_weight=0.2,
        fold_kwargs={
            "n_heads": 2,
            "n_layers": 1,
            "n_cand_layers": 1,
            "dropout": 0.0,
        },
        unfolded_kwargs={
            "n_heads": 2,
            "n_layers": 1,
            "n_time_freqs": 4,
            "dropout": 0.0,
        },
    )
    output = model(_batch(hidden=True))
    assert output["loss"].device.type == "cpu"
    assert torch.isfinite(output["loss"])
    assert "rank_target_entropy" in output
    assert "loss_rank_kl" in output
    assert output["loss_rank_kl"].item() >= -1e-5
    for key in ("rank_target_entropy", "loss_rank_kl"):
        assert torch.isfinite(output[key])
        assert not output[key].requires_grad
    torch.testing.assert_close(
        output["loss_rank_kl"],
        output["loss_rank"].detach() - output["rank_target_entropy"],
    )

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


@pytest.mark.parametrize("perfect, expected", [(True, 0.0), (False, 1.0)])
def test_fold_quality_abs_perfect_and_median(perfect, expected):
    target = torch.tensor([[-4.0, 0.0, 2.0, 9.0, 100.0]])
    hidden = torch.tensor([[True, True, True, True, False]])
    median = torch.nanmedian(
        target.masked_fill(~hidden, torch.nan), dim=-1
    ).values
    prediction = target if perfect else median[:, None].expand_as(target)
    prediction = prediction.clone()
    prediction[:, -1] = -1000.0

    loss_fvu, loss_rank, diagnostics = PretrainModel.fold_quality(
        prediction[:, None, :],
        torch.zeros(1, 1),
        target,
        hidden,
        torch.ones(1, 1, dtype=torch.bool),
        fvu_kind="abs",
    )
    assert loss_fvu.item() == pytest.approx(expected, abs=1e-6)
    assert diagnostics["fvu_top_median"].item() == pytest.approx(expected)
    assert diagnostics["fvu_best_median"].item() == pytest.approx(expected)
    assert loss_rank.item() == pytest.approx(0.0)
    assert diagnostics["rank_target_entropy"].item() == pytest.approx(0.0)
    assert diagnostics["loss_rank_kl"].item() == pytest.approx(0.0)


def test_fold_quality_abs_denominator_floor():
    target = torch.tensor([[-0.01, 0.0, 0.01]])
    loss_fvu, _, _ = PretrainModel.fold_quality(
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1),
        target,
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        fvu_kind="abs",
    )
    assert loss_fvu.item() == pytest.approx(0.02 / 0.3, rel=1e-6)


def test_fold_quality_sq_default_matches_original_formula():
    target = torch.tensor([[-4.0, 0.0, 2.0, 9.0, 100.0]])
    hidden = torch.tensor([[True, True, True, True, False]])
    pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0, -500.0]]])
    scores = torch.zeros(1, 1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    default = PretrainModel.fold_quality(pred, scores, target, hidden, mask)
    explicit = PretrainModel.fold_quality(
        pred, scores, target, hidden, mask, fvu_kind="sq"
    )
    torch.testing.assert_close(default[0], explicit[0], rtol=0, atol=0)
    torch.testing.assert_close(default[1], explicit[1], rtol=0, atol=0)
    for key in default[2]:
        torch.testing.assert_close(default[2][key], explicit[2][key], rtol=0, atol=0)

    counts = hidden.sum(dim=-1).float()
    mean = (target * hidden).sum(dim=-1) / counts
    denominator = (
        ((target - mean[:, None]) * hidden).square().sum(dim=-1)
    ).clamp_min(counts * 0.01)
    expected = (
        ((pred - target[:, None, :]).square() * hidden[:, None, :]).sum(dim=-1)
        / denominator[:, None]
    ).clamp(max=5.0).mean()
    torch.testing.assert_close(default[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["no_hidden", "no_valid"])
def test_fold_quality_empty_cases(case):
    target = torch.tensor([[-2.0, 0.0, 3.0]])
    hidden = torch.ones(1, 3, dtype=torch.bool)
    period_mask = torch.ones(1, 2, dtype=torch.bool)
    if case == "no_hidden":
        hidden.zero_()
    else:
        period_mask.zero_()
    pred = torch.zeros(1, 2, 3, requires_grad=True)
    scores = torch.zeros(1, 2).masked_fill(~period_mask, -torch.inf)
    scores.requires_grad_()

    loss_fvu, loss_rank, diagnostics = PretrainModel.fold_quality(
        pred, scores, target, hidden, period_mask, fvu_kind="abs"
    )
    assert loss_fvu.item() == 0.0
    assert loss_rank.item() == 0.0
    for value in diagnostics.values():
        assert torch.isfinite(value)
        assert value.item() == 0.0
        assert not value.requires_grad
    (loss_fvu + loss_rank).backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.isfinite(scores.grad).all()


def test_rank_diagnostics_exclude_invalid_objects_and_candidates():
    target = torch.tensor(
        [[-2.0, 0.0, 3.0], [-2.0, 0.0, 3.0], [-2.0, 0.0, 3.0]]
    )
    hidden = torch.tensor(
        [[True, True, True], [True, True, True], [False, False, False]]
    )
    period_mask = torch.tensor(
        [[True, True, False], [False, False, False], [True, True, True]]
    )
    pred = target[:, None, :].expand(-1, 3, -1).clone()
    scores = torch.tensor(
        [[0.0, 0.0, -torch.inf],
         [-torch.inf, -torch.inf, -torch.inf],
         [0.0, 1.0, 2.0]]
    )
    _, loss_rank, diagnostics = PretrainModel.fold_quality(
        pred, scores, target, hidden, period_mask, fvu_kind="abs"
    )
    assert loss_rank.item() == pytest.approx(math.log(2.0), abs=1e-6)
    assert diagnostics["rank_target_entropy"].item() == pytest.approx(
        math.log(2.0), abs=1e-6
    )
    assert diagnostics["loss_rank_kl"].item() == pytest.approx(0.0, abs=1e-6)
    for value in diagnostics.values():
        assert torch.isfinite(value)
        assert not value.requires_grad


def test_rank_target_entropy_handles_zero_probability():
    target = torch.tensor([[-2.0, 0.0, 3.0]])
    pred = torch.stack((target, target + 1000.0), dim=1)
    _, loss_rank, diagnostics = PretrainModel.fold_quality(
        pred,
        torch.zeros(1, 2),
        target,
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        fvu_kind="abs",
    )
    assert diagnostics["rank_target_entropy"].item() == pytest.approx(0.0)
    assert diagnostics["loss_rank_kl"].item() == pytest.approx(
        loss_rank.item(), abs=1e-6
    )
    assert torch.isfinite(diagnostics["loss_rank_kl"])


def test_fvu_kind_validation():
    with pytest.raises(ValueError, match="fvu_kind"):
        PretrainModel(fvu_kind="invalid")
    with pytest.raises(ValueError, match="fvu_kind"):
        PretrainModel.fold_quality(
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1),
            torch.zeros(1, 3),
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
            fvu_kind="invalid",
        )


def _normalization_batch(dtype=torch.float32):
    batch = {
        "mag": torch.tensor(
            [[-2.0, 0.0, 4.0, 10.0, 1234.0],
             [-4.0, 2.0, 8.0, -99.0, -999.0],
             [5.0, -3.0, 777.0, -777.0, 99.0]],
            dtype=dtype,
        ),
        "point_mask": torch.tensor(
            [[True, True, True, True, False],
             [True, True, True, False, False],
             [True, True, False, False, False]]
        ),
        "side": torch.arange(33, dtype=torch.float32).reshape(3, 11),
    }
    hidden = torch.tensor(
        [[False, False, True, True, False],
         [False, True, False, False, False],
         [True, True, False, False, False]]
    )
    return batch, hidden


def test_normalize_batch_hidden_points_do_not_affect_visible_or_scale():
    batch, hidden = _normalization_batch()
    original = normalize_batch(batch, hidden)
    changed_mag = batch["mag"].clone()
    changed_mag[hidden] = changed_mag[hidden] * -100.0 + 5000.0
    changed = normalize_batch(dict(batch, mag=changed_mag), hidden)
    visible = batch["point_mask"] & ~hidden

    torch.testing.assert_close(
        original["mag"][visible], changed["mag"][visible], rtol=0, atol=0
    )
    torch.testing.assert_close(
        original["side"][:, 10], changed["side"][:, 10], rtol=0, atol=0
    )
    torch.testing.assert_close(
        original["side"][:, 10],
        torch.tensor([0.0, math.log10(6.0), 0.0]),
    )
    assert not torch.equal(original["mag"][hidden], changed["mag"][hidden])


def test_normalize_batch_hidden_none_population_std_and_log_scale():
    batch, _ = _normalization_batch()
    result = normalize_batch(batch, hidden=None)

    for row in range(len(batch["mag"])):
        real = batch["point_mask"][row]
        expected_scale = batch["mag"][row, real].float().std(unbiased=False)
        assert result["mag"][row, real].std(unbiased=False).item() == pytest.approx(
            1.0, abs=1e-6
        )
        torch.testing.assert_close(
            result["side"][row, 10], expected_scale.log10()
        )
        torch.testing.assert_close(
            result["mag"][row, real],
            batch["mag"][row, real] / expected_scale,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_normalize_batch_does_not_modify_input(dtype):
    batch, hidden = _normalization_batch(dtype)
    before = {key: value.clone() for key, value in batch.items()}
    hidden_before = hidden.clone()
    result = normalize_batch(batch, hidden)

    assert result is not batch
    assert result["point_mask"] is batch["point_mask"]
    assert result["mag"].dtype == dtype
    assert result["mag"].data_ptr() != batch["mag"].data_ptr()
    assert result["side"].data_ptr() != batch["side"].data_ptr()
    for key in batch:
        torch.testing.assert_close(batch[key], before[key], rtol=0, atol=0)
    torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)
    torch.testing.assert_close(
        result["side"][:, :10], before["side"][:, :10], rtol=0, atol=0
    )


@pytest.mark.parametrize("use_hidden", [True, False])
def test_normalize_batch_padding_stays_zero(use_hidden):
    batch, hidden = _normalization_batch()
    result = normalize_batch(batch, hidden if use_hidden else None)
    padded = ~batch["point_mask"]
    torch.testing.assert_close(
        result["mag"][padded],
        torch.zeros_like(result["mag"][padded]),
        rtol=0,
        atol=0,
    )

    changed_mag = batch["mag"].clone()
    changed_mag[padded] = 100000.0
    changed = normalize_batch(
        dict(batch, mag=changed_mag), hidden if use_hidden else None
    )
    torch.testing.assert_close(result["mag"], changed["mag"], rtol=0, atol=0)
    torch.testing.assert_close(result["side"], changed["side"], rtol=0, atol=0)


def test_normalize_batch_no_visible_and_scale_floor():
    batch = {
        "mag": torch.tensor([[5.0, 8.0, 123.0], [7.0, 100.0, -999.0]]),
        "point_mask": torch.tensor(
            [[True, True, False], [True, True, False]]
        ),
        "side": torch.zeros(2, 11),
    }
    hidden = torch.tensor(
        [[True, True, False], [False, True, False]]
    )
    result = normalize_batch(batch, hidden, eps=0.01)
    torch.testing.assert_close(
        result["mag"], torch.tensor([[5.0, 8.0, 0.0], [700.0, 10000.0, 0.0]])
    )
    torch.testing.assert_close(
        result["side"][:, 10], torch.tensor([0.0, -2.0])
    )
    assert torch.isfinite(result["mag"]).all()
    assert torch.isfinite(result["side"]).all()


def test_pretrain_visible_normalization_cpu_forward_backward():
    torch.manual_seed(11)
    options = {
        "d_model": 16,
        "side_dim": 11,
        "fvu_kind": "abs",
        "fvu_weight": 0.1,
        "rank_weight": 0.2,
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
    model = PretrainModel(**options, normalize=True)
    batch = collate(
        [_object(1, scale_kind="none"), _object(2, scale_kind="none")]
    )
    batch["hidden"] = torch.zeros_like(batch["point_mask"])
    batch["hidden"][:, ::2] = True
    before = {key: value.clone() for key, value in batch.items()}
    normalized = normalize_batch(batch, batch["hidden"])

    changed_hidden_mag = batch["mag"].clone()
    changed_hidden_mag[batch["hidden"]] += 100.0
    changed_hidden = normalize_batch(
        dict(batch, mag=changed_hidden_mag), batch["hidden"]
    )
    visible = batch["point_mask"] & ~batch["hidden"]
    torch.testing.assert_close(
        normalized["mag"][visible], changed_hidden["mag"][visible], rtol=0, atol=0
    )
    torch.testing.assert_close(
        normalized["side"][:, 10], changed_hidden["side"][:, 10], rtol=0, atol=0
    )

    output = model(batch)
    assert model.normalize is True
    assert output["loss"].device.type == "cpu"
    assert torch.isfinite(output["loss"])
    assert output["loss_rank_kl"].item() >= -1e-5

    reference = PretrainModel(**options, normalize=False)
    reference.load_state_dict(model.state_dict())
    with torch.no_grad():
        expected = reference(normalized)
        changed_visible_mag = batch["mag"].clone()
        changed_visible_mag[0, 1] += 0.75
        changed_output = model(dict(batch, mag=changed_visible_mag))
    torch.testing.assert_close(output["loss"], expected["loss"])
    assert torch.isfinite(changed_output["loss"])
    assert not torch.isclose(output["loss"].detach(), changed_output["loss"])

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
        torch.testing.assert_close(batch[key], before[key], rtol=0, atol=0)


def test_build_object_none_preserves_centred_magnitudes():
    obj = _object(scale_kind="none")
    expected = np.array(
        [-2.0, 0.0, 2.0, -1.0, 0.0, 1.0, -3.0, 0.0, 3.0],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(obj["mag"], expected)
    assert obj["side"][10] == 0.0
    np.testing.assert_array_equal(
        obj["side"][:10],
        [2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    )


def test_none_scale_rejects_asinh():
    with pytest.raises(ValueError, match="mag_transform"):
        _object(scale_kind="none", mag_transform="asinh")
    with pytest.raises(ValueError, match="mag_transform"):
        LightCurveDataset(
            [], "unused", scale_kind="none", mag_transform="asinh"
        )


def test_build_cache_none_metadata_and_cli(tmp_path, monkeypatch):
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    _rows().to_parquet(lc_dir / "part.parquet", index=False)
    cands_path = tmp_path / "cands.parquet"
    pd.DataFrame(
        {"_id": [1], "period_1_LS": [1.0], "period_2_LS": [2.3]}
    ).to_parquet(cands_path, index=False)
    out = tmp_path / "cache"

    meta = build_cache.build_cache(
        lc_dir,
        cands_path,
        out,
        min_points=1,
        spike_floor=None,
        min_cadence_minutes=None,
        mag_transform=None,
        scale_kind="none",
    )
    assert meta["parameters"]["scale_kind"] == "none"
    assert meta["parameters"]["mag_transform"] is None
    assert json.loads((out / "meta.json").read_text(encoding="utf-8")) == meta
    item = CachedLightCurveDataset(out, train=False)[0]
    expected = _object(scale_kind="none")
    np.testing.assert_array_equal(item["mag"], expected["mag"])
    np.testing.assert_array_equal(item["side"], expected["side"])

    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(build_cache, "build_cache", capture)
    build_cache.main(
        ["--lc", "lc", "--cands", "cands", "--out", "out",
         "--scale-kind", "none", "--mag-transform", "none"]
    )
    assert calls[-1]["scale_kind"] == "none"
    assert calls[-1]["mag_transform"] is None
