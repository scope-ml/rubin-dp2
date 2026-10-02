# dataset.py
"""Light-curve dataset for the DP2 classifier.

Each object is turned into:
  * the filtered light curve (time since first point, band index, normalised magnitude), and
  * its de-duplicated list of candidate periods from the period search (LS, CE, AOV, FPW, MHF,
    ranks 1-50), with the number of cycles each period covers over the baseline.

Preprocessing, in order:
  1. per-band spike filter (a point more than `spike_floor` mag from the median of itself and its
     two neighbours is removed; the first/last point is compared with its single neighbour),
  2. per-band cadence filter (points closer than `min_cadence_minutes` to the previous kept point
     in the same band are dropped),
  3. per-band median subtraction (removes colour offsets between bands),
  4. division by one robust scale shared by all bands (1.4826 x median absolute value), so
     relative amplitudes between bands are preserved,
  5. optional asinh compression, which keeps small variations linear but limits the influence of
     very large excursions (deep eclipses, flares).
"""
from glob import glob
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


BANDS = ("u", "g", "r", "i", "z", "y")
# Candidate-period columns in the period-search output: period_<rank>_<algorithm>.
PERIOD_COLUMN = re.compile(r"period_(?:[1-9]|[1-4][0-9]|50)_(?:LS|CE|AOV|FPW|MHF)$")


def _filter_points(mjd, band, mag, spike_floor, min_cadence_minutes):
    """Return a mask after per-band spike filtering, then cadence filtering."""
    keep = np.ones(len(mjd), dtype=bool)

    for band_index in np.unique(band):
        indices = np.flatnonzero(band == band_index).tolist()

        # Spike filter: walk through the band's points in time order and drop a point that
        # deviates from the median of (previous, itself, next) by more than spike_floor. After a
        # removal the same position is re-checked against its new neighbours.
        if spike_floor is not None and len(indices) >= 3:
            survivors = indices.copy()
            position = 1
            while position < len(survivors) - 1:
                left, center, right = survivors[position - 1 : position + 2]
                local_median = np.median((mag[left], mag[center], mag[right]))
                if abs(mag[center] - local_median) > spike_floor:
                    survivors.pop(position)
                else:
                    position += 1

            # End points have only one neighbour, so they are compared with it directly.
            if len(survivors) >= 2:
                remove_first = (
                    abs(mag[survivors[0]] - mag[survivors[1]]) > spike_floor
                )
                remove_last = (
                    abs(mag[survivors[-1]] - mag[survivors[-2]]) > spike_floor
                )
                if remove_first:
                    survivors.pop(0)
                if remove_last:
                    survivors.pop()

            keep[indices] = False
            keep[survivors] = True

        # Cadence filter: within a band, keep a point only if it is at least
        # min_cadence_minutes after the previously kept point.
        if min_cadence_minutes is not None and min_cadence_minutes > 0:
            cadence_days = min_cadence_minutes / 1440.0
            surviving = np.flatnonzero(keep & (band == band_index))
            if len(surviving):
                last_kept = surviving[0]
                for current in surviving[1:]:
                    if mjd[current] - mjd[last_kept] < cadence_days:
                        keep[current] = False
                    else:
                        last_kept = current

    return keep


def build_object(
    lc_rows,
    cand_row,
    min_points=20,
    *,
    spike_floor=1.0,
    min_cadence_minutes=5.0,
    mag_transform="asinh",
):
    """Build one object from its light-curve rows and candidate-period row.

    Returns None if fewer than `min_points` points survive the filters or if the object has no
    valid candidate period.
    """
    if mag_transform not in ("asinh", None):
        raise ValueError("mag_transform must be 'asinh' or None")
    if len(lc_rows) == 0:
        return None

    band = pd.Categorical(lc_rows["band"], categories=BANDS).codes.astype(np.int64)
    if np.any(band < 0):
        raise ValueError("Light curve contains an unknown band")

    mjd = lc_rows["mjd"].to_numpy(dtype=np.float64)
    raw_mag = lc_rows["mag"].to_numpy(dtype=np.float64)
    order = np.argsort(mjd, kind="stable")
    mjd, band, raw_mag = mjd[order], band[order], raw_mag[order]

    keep = _filter_points(
        mjd, band, raw_mag, spike_floor, min_cadence_minutes
    )
    mjd, band, raw_mag = mjd[keep], band[keep], raw_mag[keep]
    if len(mjd) < min_points or len(mjd) == 0:
        return None

    # Time is measured from the first kept point.
    t = mjd - mjd[0]
    baseline = float(t[-1])

    # Subtract each band's median, then divide by one robust scale shared by all bands.
    centered = raw_mag.copy()
    for band_index in np.unique(band):
        selected = band == band_index
        centered[selected] -= np.median(raw_mag[selected])

    scale = max(1.4826 * np.median(np.abs(centered)), 1e-3)
    normalized_mag = centered / scale
    if mag_transform == "asinh":
        normalized_mag = np.arcsinh(normalized_mag)

    # Collect all candidate periods (a pandas row with named columns, or a plain array).
    if hasattr(cand_row, "items"):
        values = [
            value for name, value in cand_row.items()
            if PERIOD_COLUMN.fullmatch(str(name))
        ]
    else:
        values = cand_row

    periods = np.asarray(values, dtype=np.float64)
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if len(periods) == 0:
        return None

    # Merge candidates that the baseline cannot distinguish: frequencies closer than half a
    # frequency-resolution element (0.5 / baseline) are treated as the same candidate.
    frequencies = np.sort(1.0 / periods)
    resolution = 0.5 / max(baseline, 1.0)
    kept = [frequencies[0]]
    for frequency in frequencies[1:]:
        if frequency - kept[-1] > resolution:
            kept.append(frequency)

    periods = (1.0 / np.asarray(kept)).astype(np.float32)
    return {
        "t": t.astype(np.float32),
        "band": band,
        "mag": normalized_mag.astype(np.float32),
        "baseline": np.float32(baseline),
        "periods": periods,
        # Number of cycles each candidate period covers over the observed baseline.
        "cycles": (baseline / periods).astype(np.float32),
        "id": np.int64(lc_rows["diaObjectId"].iloc[0]),
    }


def _light_curve_paths(lc_paths):
    """Expand a file, a directory of parquet files, a glob pattern, or a list of these."""
    if isinstance(lc_paths, (str, Path)):
        lc_paths = [lc_paths]
    paths = []
    for entry in lc_paths:
        entry = str(entry)
        if any(character in entry for character in "*?["):
            paths.extend(Path(path) for path in sorted(glob(entry)))
        elif Path(entry).is_dir():
            paths.extend(sorted(Path(entry).glob("*.parquet")))
        else:
            paths.append(Path(entry))
    return paths


class LightCurveDataset(Dataset):
    """In-memory dataset built directly from light-curve parquet files and the candidate table.

    For training, each item is a random subset of at most `max_points` points and `max_periods`
    candidate periods (a different subset each epoch, reproducible from seed, epoch and index).
    For evaluation the subsets are deterministic (evenly spaced points, the first candidates).
    For the full 327k-object run the preprocessed cache in cached_dataset.py is used instead.
    """

    def __init__(
        self,
        lc_paths,
        cands_path,
        max_points=1024,
        max_periods=256,
        min_points=20,
        train=True,
        seed=0,
        *,
        spike_floor=1.0,
        min_cadence_minutes=5.0,
        mag_transform="asinh",
    ):
        if max_points < 1 or max_periods < 1 or min_points < 1:
            raise ValueError("max_points, max_periods, and min_points must be positive")
        if mag_transform not in ("asinh", None):
            raise ValueError("mag_transform must be 'asinh' or None")

        self.max_points = max_points
        self.max_periods = max_periods
        self.train = train
        self.seed = seed
        self.mag_transform = mag_transform
        self.epoch = 0
        self.objects = []

        # Candidate periods keyed by object id.
        candidates = pd.read_parquet(cands_path)
        period_columns = [
            column for column in candidates.columns
            if PERIOD_COLUMN.fullmatch(str(column))
        ]
        candidate_rows = {
            int(object_id): values
            for object_id, values in zip(
                candidates["_id"].to_numpy(),
                candidates[period_columns].to_numpy(dtype=np.float64),
            )
        }

        # Build every object that has both a light curve and candidate periods.
        for path in _light_curve_paths(lc_paths):
            frame = pd.read_parquet(
                path, columns=["diaObjectId", "mjd", "band", "mag"]
            )
            for object_id, rows in frame.groupby("diaObjectId", sort=False):
                candidate_row = candidate_rows.get(int(object_id))
                if candidate_row is None:
                    continue
                obj = build_object(
                    rows,
                    candidate_row,
                    min_points=min_points,
                    spike_floor=spike_floor,
                    min_cadence_minutes=min_cadence_minutes,
                    mag_transform=mag_transform,
                )
                if obj is not None:
                    self.objects.append(obj)

    def set_epoch(self, epoch):
        """Change the epoch so training subsets differ between epochs."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        obj = self.objects[index]
        point_count = len(obj["t"])
        period_count = len(obj["periods"])

        if self.train:
            # Random subsets, reproducible from (seed, epoch, index); order is kept by sorting.
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
            # Deterministic subsets for evaluation: evenly spaced points, first candidates.
            point_indices = (
                np.linspace(
                    0, point_count - 1, self.max_points, dtype=np.int64
                )
                if point_count > self.max_points
                else np.arange(point_count)
            )
            period_indices = np.arange(min(period_count, self.max_periods))

        return {
            "t": obj["t"][point_indices],
            "band": obj["band"][point_indices],
            "mag": obj["mag"][point_indices],
            "baseline": obj["baseline"],
            "periods": obj["periods"][period_indices],
            "cycles": obj["cycles"][period_indices],
            "id": obj["id"],
        }


def collate(batch):
    """Pad a list of objects into batch tensors.

    Points and candidate periods are padded to the longest object in the batch;
    `point_mask` and `period_mask` mark the real (non-padded) entries. Padded periods are set to
    1.0 (not 0) so that phase computations never divide by zero.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(batch)
    max_points = max(len(obj["t"]) for obj in batch)
    max_periods = max(len(obj["periods"]) for obj in batch)

    result = {
        "t": torch.zeros((batch_size, max_points), dtype=torch.float32),
        "band": torch.zeros((batch_size, max_points), dtype=torch.int64),
        "mag": torch.zeros((batch_size, max_points), dtype=torch.float32),
        "point_mask": torch.zeros((batch_size, max_points), dtype=torch.bool),
        "periods": torch.ones((batch_size, max_periods), dtype=torch.float32),
        "cycles": torch.zeros((batch_size, max_periods), dtype=torch.float32),
        "period_mask": torch.zeros((batch_size, max_periods), dtype=torch.bool),
        "baseline": torch.empty(batch_size, dtype=torch.float32),
        "id": torch.empty(batch_size, dtype=torch.int64),
    }

    for row, obj in enumerate(batch):
        n, k = len(obj["t"]), len(obj["periods"])
        result["t"][row, :n] = torch.as_tensor(obj["t"])
        result["band"][row, :n] = torch.as_tensor(obj["band"])
        result["mag"][row, :n] = torch.as_tensor(obj["mag"])
        result["point_mask"][row, :n] = True
        result["periods"][row, :k] = torch.as_tensor(obj["periods"])
        result["cycles"][row, :k] = torch.as_tensor(obj["cycles"])
        result["period_mask"][row, :k] = True
        result["baseline"][row] = float(obj["baseline"])
        result["id"][row] = int(obj["id"])

    return result
