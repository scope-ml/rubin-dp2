# test_dataset.py
import numpy as np
import pandas as pd
import torch

from dataset import LightCurveDataset, build_object, collate


OBJECT_KEYS = {"t", "band", "mag", "baseline", "periods", "cycles", "id", "side"}


def curve(object_id, times, bands=None, mags=None):
    times = np.asarray(times, dtype=np.float64)
    if bands is None:
        bands = ["r"] * len(times)
    if mags is None:
        mags = 20 + np.sin(times)
    return pd.DataFrame(
        {
            "diaObjectId": np.full(len(times), object_id, dtype=np.int64),
            "mjd": times,
            "band": bands,
            "mag": np.asarray(mags, dtype=np.float32),
        }
    )


def write_parquet_inputs(tmp_path, curves, candidate_rows):
    lc_path = tmp_path / "part_0000.parquet"
    cands_path = tmp_path / "cands.parquet"
    pd.concat(curves, ignore_index=True).sample(frac=1, random_state=7).to_parquet(
        lc_path, index=False
    )
    pd.DataFrame(candidate_rows).to_parquet(cands_path, index=False)
    return [lc_path], cands_path


def test_per_band_normalization():
    times = np.linspace(0, 10, 40)
    rows = pd.concat(
        [
            curve(1, times, [band] * len(times), offset + np.sin(times))
            for band, offset in [("g", 18.0), ("r", 22.0)]
        ],
        ignore_index=True,
    )
    obj = build_object(rows, {"period_1_LS": 2.0})
    g = obj["mag"][obj["band"] == 1]
    r = obj["mag"][obj["band"] == 2]
    np.testing.assert_allclose(g, r, atol=1e-5)


def test_period_merging_and_no_algorithm_or_rank_leakage():
    rows = curve(3, np.linspace(0, 10, 20))
    obj = build_object(
        rows,
        {
            "period_1_LS": 1.0,
            "period_2_CE": 1.0001,
            "period_3_AOV": 2.0,
            "period_4_FPW": 0.5,
            "period_5_MHF": np.nan,
            "period_6_LS": -1.0,
        },
    )
    assert set(obj) == OBJECT_KEYS
    np.testing.assert_allclose(obj["periods"], [2.0, 1.0001, 1.0, 0.5], rtol=1e-6)
    np.testing.assert_allclose(obj["cycles"], 10 / obj["periods"])


def test_dataset_subsets_are_reproducible(tmp_path):
    periods = {f"period_{i + 1}_LS": 0.1 + i for i in range(30)}
    paths, cands = write_parquet_inputs(
        tmp_path,
        [curve(10, np.linspace(0, 100, 100))],
        [{"_id": 10, **periods}],
    )
    train = LightCurveDataset(paths, cands, max_points=20, max_periods=5, seed=23)
    train.set_epoch(0)
    first = train[0]
    again = train[0]
    for key in OBJECT_KEYS:
        np.testing.assert_array_equal(first[key], again[key])

    train.set_epoch(1)
    next_epoch = train[0]
    assert not np.array_equal(first["t"], next_epoch["t"])
    train.set_epoch(0)
    np.testing.assert_array_equal(train[0]["t"], first["t"])
    np.testing.assert_array_equal(train[0]["periods"], first["periods"])

    evaluation = LightCurveDataset(
        paths, cands, max_points=20, max_periods=5, train=False
    )
    before = evaluation[0]
    evaluation.set_epoch(99)
    after = evaluation[0]
    for key in OBJECT_KEYS:
        np.testing.assert_array_equal(before[key], after[key])


def test_collate_shapes_masks_and_padding():
    a = build_object(
        curve(1, np.linspace(0, 10, 5)),
        {"period_1_LS": 1.0},
        min_points=1,
        spike_floor=None,
        min_cadence_minutes=None,
    )
    b = build_object(
        curve(2, np.linspace(0, 10, 3)),
        {"period_1_LS": 1.0, "period_2_CE": 2.0},
        min_points=1,
        spike_floor=None,
        min_cadence_minutes=None,
    )
    batch = collate([a, b])
    assert batch["t"].shape == (2, 5)
    assert batch["band"].shape == (2, 5)
    assert batch["mag"].shape == (2, 5)
    assert batch["periods"].shape == (2, 2)
    assert batch["cycles"].shape == (2, 2)
    assert batch["baseline"].shape == (2,)
    assert batch["id"].shape == (2,)
    assert batch["point_mask"].tolist() == [[True] * 5, [True] * 3 + [False] * 2]
    assert batch["period_mask"].tolist() == [[True, False], [True, True]]
    assert torch.equal(batch["t"][1, 3:], torch.zeros(2))
    assert torch.equal(batch["band"][1, 3:], torch.zeros(2, dtype=torch.int64))
    assert torch.equal(batch["mag"][1, 3:], torch.zeros(2))
    assert batch["periods"][0, 1].item() == 1.0
    assert batch["cycles"][0, 1].item() == 0.0
    assert batch["point_mask"].dtype == torch.bool
    assert batch["period_mask"].dtype == torch.bool


def test_short_objects_are_skipped(tmp_path):
    paths, cands = write_parquet_inputs(
        tmp_path,
        [
            curve(1, np.arange(19)),
            curve(2, np.arange(20)),
        ],
        [
            {"_id": 1, "period_1_LS": 2.0},
            {"_id": 2, "period_1_LS": 2.0},
        ],
    )
    dataset = LightCurveDataset(paths, cands)
    assert len(dataset) == 1
    assert dataset[0]["id"] == 2


def test_twenty_point_single_band_object_end_to_end(tmp_path):
    paths, cands = write_parquet_inputs(
        tmp_path,
        [curve(42, np.linspace(0, 8, 20))],
        [{"_id": 42, "period_1_CE": 1.5}],
    )
    dataset = LightCurveDataset(paths, cands, train=False)
    assert len(dataset) == 1
    obj = dataset[0]
    assert len(obj["t"]) == 20
    assert np.all(obj["band"] == 2)
    assert np.all(np.diff(obj["t"]) >= 0)
    batch = collate([obj])
    assert batch["point_mask"].all()
    assert batch["period_mask"].all()
    assert batch["id"].item() == 42


def test_single_spike_removed_and_disabling_spike_filter():
    rows = curve(1, np.arange(7), mags=[20, 20, 20, 23, 20, 20, 20])
    filtered = build_object(rows, {"period_1_LS": 2.0}, min_points=1)
    unfiltered = build_object(
        rows, {"period_1_LS": 2.0}, min_points=1, spike_floor=None
    )
    np.testing.assert_array_equal(filtered["t"], [0, 1, 2, 4, 5, 6])
    assert len(unfiltered["t"]) == 7


def test_monotonic_ramp_is_not_a_spike():
    rows = curve(1, np.arange(6), mags=np.arange(6))
    obj = build_object(rows, {"period_1_LS": 2.0}, min_points=1)
    np.testing.assert_array_equal(obj["t"], np.arange(6))


def test_adjacent_spikes_use_the_surviving_pool():
    rows = curve(1, np.arange(8), mags=[0, 0, 0, 3, 6, 0, 0, 0])
    obj = build_object(rows, {"period_1_LS": 2.0}, min_points=1)
    # The +3 point is the median of (0, 3, 6), so it survives.
    # The +6 point is then removed against (3, 6, 0).
    np.testing.assert_array_equal(obj["t"], [0, 1, 2, 3, 5, 6, 7])


def test_first_and_last_spikes_are_removed():
    rows = curve(1, np.arange(6), mags=[3, 0, 0, 0, 0, 3])
    obj = build_object(rows, {"period_1_LS": 2.0}, min_points=1)
    np.testing.assert_array_equal(obj["t"], [0, 1, 2, 3])


def test_cadence_uses_last_kept_point():
    minutes = np.array([0, 2, 4, 6, 12])
    rows = curve(1, minutes / 1440, mags=np.zeros(5))
    filtered = build_object(rows, {"period_1_LS": 1.0}, min_points=1)
    unfiltered = build_object(
        rows, {"period_1_LS": 1.0}, min_points=1, min_cadence_minutes=0
    )
    np.testing.assert_allclose(filtered["t"], [0, 6 / 1440, 12 / 1440])
    assert len(unfiltered["t"]) == 5


def test_cadence_is_independent_per_band():
    rows = curve(
        1,
        np.array([0, 2, 4, 6, 12]) / 1440,
        bands=["g", "r", "g", "r", "g"],
        mags=np.zeros(5),
    )
    obj = build_object(rows, {"period_1_LS": 1.0}, min_points=1)
    np.testing.assert_allclose(obj["t"], [0, 2 / 1440, 12 / 1440])
    np.testing.assert_array_equal(obj["band"], [1, 2, 1])


def test_filters_recompute_time_baseline_cycles_and_min_points(tmp_path):
    minutes = np.array([0, 2, 8, 14, 15])
    rows = curve(9, minutes / 1440, mags=[3, 0, 0, 0, 0])
    obj = build_object(rows, {"period_1_LS": 1.0}, min_points=3)
    np.testing.assert_allclose(obj["t"], [0, 6 / 1440, 12 / 1440])
    np.testing.assert_allclose(obj["baseline"], 12 / 1440)
    np.testing.assert_allclose(obj["cycles"], obj["baseline"] / obj["periods"])
    assert build_object(rows, {"period_1_LS": 1.0}, min_points=4) is None

    paths, cands = write_parquet_inputs(
        tmp_path, [rows], [{"_id": 9, "period_1_LS": 1.0}]
    )
    dataset = LightCurveDataset(paths, cands, min_points=3, train=False)
    assert len(dataset) == 1
    np.testing.assert_allclose(dataset[0]["baseline"], 12 / 1440)
    assert len(LightCurveDataset(paths, cands, min_points=4)) == 0
