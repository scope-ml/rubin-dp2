# cached_dataset.py
"""Dataset that reads the preprocessed cache written by build_cache.py.

The cache stores every object already filtered and normalised (see dataset.build_object) as
flat arrays in .npz shards, with offset arrays giving where each object's points and candidate
periods start. Loading it avoids repeating the preprocessing for all 327k objects every run.
Item sampling is identical to dataset.LightCurveDataset.
"""
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class CachedLightCurveDataset(Dataset):
    """All cached objects in memory as flat arrays, indexed through offset arrays."""

    def __init__(
        self,
        cache_dir,
        max_points=1024,
        max_periods=256,
        train=True,
        seed=0,
    ):
        if max_points < 1 or max_periods < 1:
            raise ValueError("max_points and max_periods must be positive")

        cache_dir = Path(cache_dir)
        self.meta = json.loads(
            (cache_dir / "meta.json").read_text(encoding="utf-8")
        )
        paths = sorted(cache_dir.glob("shard_*.npz"))
        if len(paths) != self.meta["counts"]["shards"]:
            raise ValueError("Cache shard count does not match meta.json")

        # Concatenate every shard's flat arrays; shift each shard's offsets so they index into
        # the concatenated arrays.
        pieces = {
            key: [] for key in
            ("t", "band", "mag", "periods", "cycles", "id", "baseline")
        }
        point_offsets = [0]
        period_offsets = [0]
        for path in paths:
            with np.load(path, allow_pickle=False) as shard:
                for key in pieces:
                    pieces[key].append(shard[key])
                point_offsets.extend(
                    (shard["point_offsets"][1:] + point_offsets[-1]).tolist()
                )
                period_offsets.extend(
                    (
                        shard["period_offsets"][1:] + period_offsets[-1]
                    ).tolist()
                )

        dtypes = {
            "t": np.float32,
            "band": np.int8,
            "mag": np.float32,
            "periods": np.float32,
            "cycles": np.float32,
            "id": np.int64,
            "baseline": np.float32,
        }
        for key, arrays in pieces.items():
            setattr(
                self,
                key if key != "id" else "ids",
                np.concatenate(arrays)
                if arrays else np.empty(0, dtype=dtypes[key]),
            )
        self.point_offsets = np.asarray(point_offsets, dtype=np.int64)
        self.period_offsets = np.asarray(period_offsets, dtype=np.int64)
        self.max_points = max_points
        self.max_periods = max_periods
        self.train = train
        self.seed = seed
        self.epoch = 0

        # Guard against a partial or corrupted cache.
        if len(self.ids) != self.meta["counts"]["objects"]:
            raise ValueError("Cache object count does not match meta.json")
        if len(self.t) != self.meta["counts"]["points"]:
            raise ValueError("Cache point count does not match meta.json")
        if len(self.periods) != self.meta["counts"]["periods"]:
            raise ValueError("Cache period count does not match meta.json")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        point_start = self.point_offsets[index]
        point_end = self.point_offsets[index + 1]
        period_start = self.period_offsets[index]
        period_end = self.period_offsets[index + 1]
        point_count = int(point_end - point_start)
        period_count = int(period_end - period_start)

        # Same subset sampling as dataset.LightCurveDataset: random per (seed, epoch, index) for
        # training, deterministic for evaluation.
        if self.train:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, index])
            )
            point_indices = np.sort(
                rng.choice(
                    point_count,
                    size=min(point_count, self.max_points),
                    replace=False,
                )
            )
            period_indices = np.sort(
                rng.choice(
                    period_count,
                    size=min(period_count, self.max_periods),
                    replace=False,
                )
            )
        else:
            point_indices = (
                np.linspace(
                    0, point_count - 1, self.max_points, dtype=np.int64
                )
                if point_count > self.max_points
                else np.arange(point_count)
            )
            period_indices = np.arange(min(period_count, self.max_periods))

        return {
            "t": self.t[point_start:point_end][point_indices],
            "band": self.band[point_start:point_end][point_indices].astype(
                np.int64
            ),
            "mag": self.mag[point_start:point_end][point_indices],
            "baseline": self.baseline[index],
            "periods": self.periods[
                period_start:period_end
            ][period_indices],
            "cycles": self.cycles[
                period_start:period_end
            ][period_indices],
            "id": self.ids[index],
        }
