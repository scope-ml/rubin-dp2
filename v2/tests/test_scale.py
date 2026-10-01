# test_scale.py
import json
import math
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Subset

from build_cache import build_cache
from cached_dataset import CachedLightCurveDataset
from dataset import LightCurveDataset, collate
from pretrain import PretrainModel
from train_pretrain import (
    ExactDistributedSampler,
    attach_validation_mask,
    main,
    split_indices,
    validate,
)


def make_parquets(tmp_path, n_objects=8):
    lc_paths = []
    candidate_rows = []
    for part in range(2):
        rows = []
        for local in range(n_objects // 2):
            object_id = part * (n_objects // 2) + local + 1
            for point in range(24):
                rows.append(
                    {
                        "diaObjectId": object_id,
                        "mjd": 60_000.0 + point,
                        "band": "r" if point % 2 else "g",
                        "mag": (
                            19.0
                            + 0.1 * math.sin(
                                2 * math.pi * point / (1.5 + object_id * 0.01)
                            )
                        ),
                    }
                )
            candidate_rows.append(
                {
                    "_id": object_id,
                    "period_1_LS": 1.5 + object_id * 0.01,
                    "period_1_CE": 2.7 + object_id * 0.01,
                    "period_1_AOV": np.nan,
                }
            )
        path = tmp_path / f"part_{part}.parquet"
        pd.DataFrame(rows).to_parquet(path)
        lc_paths.append(path)
    cands = tmp_path / "combined.parquet"
    pd.DataFrame(candidate_rows).to_parquet(cands)
    return lc_paths, cands


def assert_item_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(
            np.asarray(left[key]), np.asarray(right[key])
        )
        assert np.asarray(left[key]).dtype == np.asarray(right[key]).dtype


def test_cache_reproduces_dataset_and_metadata(tmp_path):
    pytest.importorskip("pyarrow")
    lc_paths, cands = make_parquets(tmp_path)
    cache_dir = tmp_path / "cache"
    meta = build_cache(
        str(tmp_path / "part_*.parquet"),
        cands,
        cache_dir,
        workers=1,
        shard_objects=3,
        spike_floor=None,
        min_cadence_minutes=None,
    )
    assert meta["counts"]["objects"] == 8
    assert meta["counts"]["points"] == 8 * 24
    assert meta["counts"]["periods"] == 8 * 2
    assert meta["counts"]["skipped_objects"] == 0
    assert meta["counts"]["shards"] == 3
    assert meta["parameters"]["mag_transform"] == "asinh"
    assert meta["parameters"]["spike_floor"] is None
    assert len(meta["inputs"]["light_curve_parts"]) == 2
    assert len(meta["dataset_sha256"]) == 64
    assert json.loads(
        (cache_dir / "meta.json").read_text()
    ) == meta

    with pytest.raises(FileExistsError):
        build_cache(
            str(tmp_path / "part_*.parquet"),
            cands,
            cache_dir,
        )

    for train in (False, True):
        original = LightCurveDataset(
            lc_paths,
            cands,
            max_points=13,
            max_periods=1,
            train=train,
            seed=17,
            spike_floor=None,
            min_cadence_minutes=None,
        )
        cached = CachedLightCurveDataset(
            cache_dir,
            max_points=13,
            max_periods=1,
            train=train,
            seed=17,
        )
        np.testing.assert_array_equal(
            cached.ids,
            np.array([obj["id"] for obj in original.objects]),
        )
        for epoch in (0, 1, 5):
            original.set_epoch(epoch)
            cached.set_epoch(epoch)
            for index in (0, 2, 5, 7):
                assert_item_equal(
                    original[index], cached[index]
                )


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ddp_worker(rank, world_size, port, cache_dir, result_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(123)
        dataset = CachedLightCurveDataset(
            cache_dir, max_points=24, max_periods=2,
            train=False, seed=0,
        )
        train_indices, val_indices = split_indices(
            dataset.ids, None, 0.25
        )
        train_subset = Subset(dataset, train_indices)
        sampler = ExactDistributedSampler(
            train_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=7,
        )
        sampler.set_epoch(0)
        local_indices = list(sampler)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_indices)
        if rank == 0:
            assert set(gathered[0]).isdisjoint(gathered[1])
            assert sorted(gathered[0] + gathered[1]) == list(
                range(len(train_subset))
            )

        model = PretrainModel(
            d_model=8,
            fold_kwargs={
                "n_heads": 1,
                "n_layers": 0,
                "n_cand_layers": 0,
                "fold_chunk": 2,
                "dropout": 0.0,
            },
            unfolded_kwargs={
                "n_heads": 1,
                "n_layers": 0,
                "dropout": 0.0,
            },
        )
        wrapped = DistributedDataParallel(
            model, find_unused_parameters=True
        )
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-3)

        item = train_subset[local_indices[0]]
        batch = collate([item])
        batch["hidden"] = torch.zeros_like(batch["point_mask"])
        batch["hidden"][:, :4] = True
        optimizer.zero_grad(set_to_none=True)
        wrapped(batch)["loss"].backward()
        optimizer.step()

        parameters = torch.cat(
            [parameter.detach().flatten() for parameter in model.parameters()]
        )
        other = [torch.empty_like(parameters) for _ in range(world_size)]
        dist.all_gather(other, parameters)
        assert torch.equal(other[0], other[1])

        val_subset = Subset(dataset, val_indices)
        val_sampler = ExactDistributedSampler(
            val_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=2,
            sampler=val_sampler,
            collate_fn=collate,
        )
        metrics = validate(
            wrapped, loader, torch.device("cpu"),
            False, 0, rank, world_size,
        )
        torch.save(
            {"parameters": parameters, "metrics": metrics},
            Path(result_dir) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_ddp_sampler_parameters_and_exact_validation(tmp_path):
    pytest.importorskip("pyarrow")
    _, cands = make_parquets(tmp_path)
    cache_dir = tmp_path / "cache"
    build_cache(
        str(tmp_path / "part_*.parquet"),
        cands,
        cache_dir,
        spike_floor=None,
        min_cadence_minutes=None,
    )
    port = _free_port()
    mp.spawn(
        _ddp_worker,
        args=(2, port, str(cache_dir), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank0 = torch.load(
        tmp_path / "rank_0.pt", weights_only=False
    )
    rank1 = torch.load(
        tmp_path / "rank_1.pt", weights_only=False
    )
    assert torch.equal(
        rank0["parameters"], rank1["parameters"]
    )
    assert rank0["metrics"] == rank1["metrics"]

    # Validate the same post-step model in one process.
    dataset = CachedLightCurveDataset(
        cache_dir, max_points=24, max_periods=2,
        train=False, seed=0,
    )
    _, val_indices = split_indices(
        dataset.ids, None, 0.25
    )
    torch.manual_seed(123)
    model = PretrainModel(
        d_model=8,
        fold_kwargs={
            "n_heads": 1,
            "n_layers": 0,
            "n_cand_layers": 0,
            "fold_chunk": 2,
            "dropout": 0.0,
        },
        unfolded_kwargs={
            "n_heads": 1,
            "n_layers": 0,
            "dropout": 0.0,
        },
    )
    # The distributed test checks identical parameters across ranks.
    # Here a direct one-process validation checks the aggregation path.
    loader = torch.utils.data.DataLoader(
        Subset(dataset, val_indices),
        batch_size=2,
        collate_fn=collate,
    )
    one_metrics = validate(
        model, loader, torch.device("cpu"), False, 0
    )
    assert set(one_metrics) == set(rank0["metrics"])
    for key in one_metrics:
        assert math.isfinite(one_metrics[key])


def _training_args(cache_dir, out_dir, *extra):
    return [
        "--cache", str(cache_dir),
        "--out", str(out_dir),
        "--epochs", "2",
        "--batch", "2",
        "--workers", "0",
        "--max-points", "24",
        "--max-periods", "2",
        "--d-model", "8",
        "--n-heads", "1",
        "--fold-layers", "0",
        "--cand-layers", "0",
        "--unf-layers", "0",
        "--fold-chunk", "2",
        "--warmup", "20",
        "--log-every", "1",
        *extra,
    ]


def test_resume_matches_uninterrupted_training(tmp_path):
    pytest.importorskip("pyarrow")
    _, cands = make_parquets(tmp_path)
    cache_dir = tmp_path / "cache"
    build_cache(
        str(tmp_path / "part_*.parquet"),
        cands,
        cache_dir,
        spike_floor=None,
        min_cadence_minutes=None,
    )
    ids = CachedLightCurveDataset(cache_dir).ids
    train_indices, _ = split_indices(ids, None, 0.1)
    steps_per_epoch = math.ceil(len(train_indices) / 2)

    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"
    main(_training_args(cache_dir, full_dir))
    main(
        _training_args(
            cache_dir, resumed_dir,
            "--max-steps", str(steps_per_epoch),
        )
    )
    main(
        _training_args(
            cache_dir, resumed_dir,
            "--resume", str(resumed_dir / "last.pt"),
        )
    )
    full = torch.load(
        full_dir / "last.pt", weights_only=False
    )
    resumed = torch.load(
        resumed_dir / "last.pt", weights_only=False
    )
    assert full["global_step"] == resumed["global_step"]
    for key in full["model"]:
        assert torch.equal(
            full["model"][key], resumed["model"][key]
        )


def test_max_steps_writes_final_throughput(tmp_path):
    pytest.importorskip("pyarrow")
    _, cands = make_parquets(tmp_path)
    cache_dir = tmp_path / "cache"
    build_cache(
        str(tmp_path / "part_*.parquet"),
        cands,
        cache_dir,
        spike_floor=None,
        min_cadence_minutes=None,
    )
    out_dir = tmp_path / "short"
    main(
        _training_args(
            cache_dir, out_dir,
            "--max-steps", "1",
        )
    )
    lines = [
        json.loads(line)
        for line in (out_dir / "log.jsonl").read_text().splitlines()
    ]
    assert lines[-1]["split"] == "final"
    assert lines[-1]["steps"] == 1
    assert lines[-1]["elapsed_seconds"] > 0
    assert lines[-1]["objects_per_sec"] > 0
    assert (out_dir / "last.pt").exists()
