# build_cache.py
"""Build the preprocessed cache used for full-scale pretraining.

Reads the light-curve parquet parts and the candidate-period table, runs
dataset.build_object on every object (filters, normalisation, candidate de-duplication), and
writes the results as flat arrays in .npz shards plus a meta.json with counts, parameters,
input file fingerprints and a hash of dataset.py (so a cache can be matched to the code that
made it). Objects that are skipped are counted by reason.

Example:
    python build_cache.py --lc lc_parts/ --cands candidates.parquet --out cache/ --workers 16
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dataset import PERIOD_COLUMN, _filter_points, build_object


# Per-worker globals, set once by _init_worker so the large candidate table is not pickled
# for every task.
_CANDIDATES = None
_BUILD_OPTIONS = None
_IDS = None


def resolve_parts(spec):
    """Light-curve parquet parts from a directory or a glob pattern, in sorted order."""
    path = Path(spec)
    if path.is_dir():
        parts = sorted(path.glob("*.parquet"))
    else:
        parts = [Path(name) for name in sorted(glob(spec))]
    if not parts:
        raise FileNotFoundError(f"No light-curve parquet parts matched {spec!r}")
    return [part.resolve() for part in parts]


def input_info(path):
    """Size and modification time of an input file, stored in meta.json for provenance."""
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def dataset_sha256():
    """Hash of dataset.py, so the cache records which preprocessing code produced it."""
    path = Path(__file__).with_name("dataset.py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _init_worker(candidates, options, ids=None):
    global _CANDIDATES, _BUILD_OPTIONS, _IDS
    _CANDIDATES = candidates
    _BUILD_OPTIONS = options
    _IDS = ids


def _skip_reason(rows, candidate_values, options):
    """Work out why build_object returned None, for the skipped-object counts."""
    if candidate_values is None:
        return "no_candidates"

    band = pd.Categorical(
        rows["band"], categories=("u", "g", "r", "i", "z", "y")
    ).codes.astype(np.int64)
    mjd = rows["mjd"].to_numpy(dtype=np.float64)
    mag = rows["mag"].to_numpy(dtype=np.float64)
    order = np.argsort(mjd, kind="stable")
    keep = _filter_points(
        mjd[order],
        band[order],
        mag[order],
        options["spike_floor"],
        options["min_cadence_minutes"],
    )
    if int(keep.sum()) < options["min_points"]:
        return "too_few_points"

    periods = np.asarray(candidate_values, dtype=np.float64)
    if not np.any(np.isfinite(periods) & (periods > 0)):
        return "no_valid_periods"
    return "other"


def _process_part(path):
    """Build all objects in one parquet part; return them with counts of kept and skipped."""
    frame = pd.read_parquet(
        path, columns=["diaObjectId", "mjd", "band", "mag"]
    )
    if _IDS is not None:
        frame = frame[frame["diaObjectId"].isin(_IDS)]
    objects = []
    counts = Counter()
    for object_id, rows in frame.groupby("diaObjectId", sort=False):
        candidate_values = _CANDIDATES.get(int(object_id))
        if candidate_values is None:
            counts["no_candidates"] += 1
            continue

        obj = build_object(
            rows,
            candidate_values,
            min_points=_BUILD_OPTIONS["min_points"],
            spike_floor=_BUILD_OPTIONS["spike_floor"],
            min_cadence_minutes=_BUILD_OPTIONS["min_cadence_minutes"],
            mag_transform=_BUILD_OPTIONS["mag_transform"],
            scale_kind=_BUILD_OPTIONS.get("scale_kind", "mad"),
        )
        if obj is None:
            counts[_skip_reason(rows, candidate_values, _BUILD_OPTIONS)] += 1
        else:
            objects.append(obj)
            counts["objects"] += 1
            counts["points"] += len(obj["t"])
            counts["periods"] += len(obj["periods"])
    return objects, dict(counts)


def _concat(objects, key, dtype):
    return np.concatenate(
        [np.asarray(obj[key], dtype=dtype) for obj in objects]
    )


def _write_shard(path, objects):
    """Write objects as flat arrays; offsets[i]:offsets[i+1] selects object i's entries."""
    point_lengths = np.fromiter(
        (len(obj["t"]) for obj in objects), dtype=np.int64
    )
    period_lengths = np.fromiter(
        (len(obj["periods"]) for obj in objects), dtype=np.int64
    )
    point_offsets = np.empty(len(objects) + 1, dtype=np.int64)
    period_offsets = np.empty(len(objects) + 1, dtype=np.int64)
    point_offsets[0] = 0
    period_offsets[0] = 0
    np.cumsum(point_lengths, out=point_offsets[1:])
    np.cumsum(period_lengths, out=period_offsets[1:])

    extra = {}
    if all("side" in obj for obj in objects):
        extra["side"] = np.stack(
            [np.asarray(obj["side"], dtype=np.float32) for obj in objects]
        )

    np.savez(
        path,
        t=_concat(objects, "t", np.float32),
        band=_concat(objects, "band", np.int8),
        mag=_concat(objects, "mag", np.float32),
        periods=_concat(objects, "periods", np.float32),
        cycles=_concat(objects, "cycles", np.float32),
        point_offsets=point_offsets,
        period_offsets=period_offsets,
        id=np.fromiter(
            (obj["id"] for obj in objects), dtype=np.int64
        ),
        baseline=np.fromiter(
            (obj["baseline"] for obj in objects), dtype=np.float32
        ),
        **extra,
    )


def build_cache(
    lc,
    cands,
    out,
    *,
    workers=1,
    shard_objects=10_000,
    min_points=20,
    spike_floor=1.0,
    min_cadence_minutes=5.0,
    mag_transform="asinh",
    scale_kind="mad",
    overwrite=False,
    ids=None,
):
    """Build the cache in `out` and return its metadata. Parameters match build_object.

    `ids`: optional CSV with a column `oid`; only those objects are built (e.g. an evaluation set).
    """
    if workers < 1 or shard_objects < 1 or min_points < 1:
        raise ValueError("workers, shard_objects and min_points must be positive")
    if mag_transform not in ("asinh", None):
        raise ValueError("mag_transform must be 'asinh' or None")
    if scale_kind not in ("mad", "std", "none"):
        raise ValueError("scale_kind must be 'mad', 'std' or 'none'")

    parts = resolve_parts(lc)
    cands_path = Path(cands).resolve()
    ids_path = None
    requested_ids = None
    if ids is not None:
        ids_path = Path(ids).resolve()
        requested_ids = {
            int(value)
            for value in pd.read_csv(
                ids_path, usecols=["oid"], dtype={"oid": np.int64}
            )["oid"]
        }
    out_dir = Path(out).resolve()
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)

    schema = pq.ParquetFile(cands_path).schema_arrow
    period_columns = [
        name for name in schema.names if PERIOD_COLUMN.fullmatch(name)
    ]
    candidate_frame = pd.read_parquet(
        cands_path, columns=["_id", *period_columns]
    )
    candidates = {
        int(object_id): values
        for object_id, values in zip(
            candidate_frame["_id"].to_numpy(),
            candidate_frame[period_columns].to_numpy(dtype=np.float64),
        )
    }
    del candidate_frame

    options = {
        "min_points": min_points,
        "spike_floor": spike_floor,
        "min_cadence_minutes": min_cadence_minutes,
        "mag_transform": mag_transform,
        "scale_kind": scale_kind,
        "candidate_periods": "exact duplicates removed",
    }
    out_dir.mkdir(parents=True)
    pending = []
    totals = Counter()
    shard_count = 0

    # Collect objects as parts finish and write a shard every `shard_objects` objects.
    def consume(result):
        nonlocal shard_count
        objects, counts = result
        totals.update(counts)
        pending.extend(objects)
        while len(pending) >= shard_objects:
            _write_shard(
                out_dir / f"shard_{shard_count:04d}.npz",
                pending[:shard_objects],
            )
            del pending[:shard_objects]
            shard_count += 1

    if workers == 1:
        _init_worker(candidates, options, requested_ids)
        for part in parts:
            consume(_process_part(part))
    else:
        # imap preserves part order, matching LightCurveDataset's object order.
        with mp.get_context("spawn").Pool(
            workers,
            initializer=_init_worker,
            initargs=(candidates, options, requested_ids),
        ) as pool:
            for result in pool.imap(_process_part, parts):
                consume(result)

    if pending:
        _write_shard(
            out_dir / f"shard_{shard_count:04d}.npz", pending
        )
        shard_count += 1

    skipped = {
        reason: int(totals[reason])
        for reason in (
            "no_candidates",
            "too_few_points",
            "no_valid_periods",
            "other",
        )
    }
    meta = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_sha256(),
        "parameters": {
            **options,
            "shard_objects": shard_objects,
        },
        "inputs": {
            "light_curve_parts": [input_info(part) for part in parts],
            "candidates": input_info(cands_path),
            "candidate_period_columns": period_columns,
        },
        "counts": {
            "objects": int(totals["objects"]),
            "points": int(totals["points"]),
            "periods": int(totals["periods"]),
            "skipped_objects": int(sum(skipped.values())),
            "skipped_by_reason": skipped,
            "shards": shard_count,
        },
    }
    if ids_path is not None:
        meta["inputs"]["ids"] = {
            **input_info(ids_path),
            "count": len(requested_ids),
        }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def _optional_float(value):
    return None if value.lower() == "none" else float(value)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc", required=True)
    parser.add_argument("--cands", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-objects", type=int, default=10_000)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--spike-floor", type=_optional_float, default=1.0)
    parser.add_argument(
        "--min-cadence-minutes", type=_optional_float, default=5.0
    )
    parser.add_argument(
        "--mag-transform", choices=("asinh", "none"), default="asinh"
    )
    parser.add_argument(
        "--scale-kind", choices=("mad", "std", "none"), default="mad"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build_cache(
        args.lc,
        args.cands,
        args.out,
        workers=args.workers,
        shard_objects=args.shard_objects,
        min_points=args.min_points,
        spike_floor=args.spike_floor,
        min_cadence_minutes=args.min_cadence_minutes,
        mag_transform=None if args.mag_transform == "none" else "asinh",
        scale_kind=args.scale_kind,
        overwrite=args.overwrite,
        ids=args.ids,
    )


if __name__ == "__main__":
    main()
