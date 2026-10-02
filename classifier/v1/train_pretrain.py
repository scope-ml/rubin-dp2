# train_pretrain.py
"""Self-supervised pretraining, single GPU or multi-GPU (torchrun, DistributedDataParallel).

Per training step: a batch of objects is sampled, 30% of each object's observing nights are
hidden (masking.night_mask, seeded by step and rank so runs are reproducible), and
pretrain.PretrainModel is trained to predict the hidden magnitudes. AdamW with linear warm-up
then cosine decay to zero over all epochs; optional bf16 autocast (--amp).

Validation: a fixed 10% of objects, chosen by a hash of the object id (so the split does not
depend on file order or on the number of GPUs), each evaluated with a fixed per-object mask.
After every epoch: last.pt, and best.pt when the validation loss improves. Checkpoints store the
model, optimizer, RNG states and position in the epoch, so --resume continues exactly.
Metrics are appended to log.jsonl.

Command used for the 327k-object run (1 node, 4 GPUs):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 train_pretrain.py         --out runs/full50 --cache cache_full --encoder rope --d-model 128 --n-heads 8         --fold-layers 4 --cand-layers 2 --unf-layers 4 --n-harm 4 --batch 32 --lr 6e-4         --weight-decay 0.05 --warmup 1000 --epochs 50 --val-frac 0.1 --seed 0         --fold-chunk 512 --amp --workers 6 --log-every 100 --mag-transform asinh         --likelihood laplace --fvu-weight 1.0 --rank-weight 1.0 --rank-tau 0.1 --fvu-cap 5.0
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, Subset

from cached_dataset import CachedLightCurveDataset
from dataset import LightCurveDataset, collate
from masking import night_mask
from pretrain import PretrainModel


METRIC_KEYS = (
    "loss",
    "loss_fold",
    "loss_unf",
    "loss_fvu",
    "loss_rank",
    "sigma",
    "cand_entropy",
    "top_is_best",
    "mse_best_fold",
    "mse_top_fold",
    "fvu_top_median",
    "fvu_best_median",
    "frac_top_fvu_lt1",
    "frac_top_fvu_lt03",
)
MEDIAN_KEYS = ("fvu_top_median", "fvu_best_median")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache")
    source.add_argument("--lc", help="Light-curve glob or directory")
    parser.add_argument("--cands")
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", choices=("rope", "cnn"), default="rope")
    parser.add_argument(
        "--mag-transform", choices=("asinh", "none"), default="asinh"
    )
    parser.add_argument(
        "--likelihood", choices=("laplace", "gaussian"), default="laplace"
    )
    parser.add_argument("--fvu-weight", type=float, default=0.0)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--rank-tau", type=float, default=0.1)
    parser.add_argument("--fvu-cap", type=float, default=5.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--fold-layers", type=int, default=4)
    parser.add_argument("--cand-layers", type=int, default=2)
    parser.add_argument("--unf-layers", type=int, default=4)
    parser.add_argument("--n-harm", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--max-periods", type=int, default=256)
    parser.add_argument("--fold-chunk", type=int, default=128)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)

    if args.cache and args.cands:
        parser.error("--cands is only valid with --lc")
    if args.lc and not args.cands:
        parser.error("--lc requires --cands")
    if args.epochs < 1 or args.batch < 1 or args.log_every < 1:
        parser.error("--epochs, --batch and --log-every must be positive")
    if args.workers < 0 or args.warmup < 0:
        parser.error("--workers and --warmup must be nonnegative")
    if not 0.0 < args.val_frac < 1.0:
        parser.error("--val-frac must be between 0 and 1")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.fvu_weight < 0 or args.rank_weight < 0:
        parser.error("--fvu-weight and --rank-weight must be nonnegative")
    if args.rank_tau <= 0 or args.fvu_cap <= 0:
        parser.error("--rank-tau and --fvu-cap must be positive")
    if (
        args.d_model < 1
        or args.n_heads < 1
        or args.d_model % args.n_heads
        or (args.d_model // args.n_heads) % 2
    ):
        parser.error("--d-model / --n-heads must be an even integer")
    if min(
        args.fold_layers, args.cand_layers, args.unf_layers
    ) < 0 or args.n_harm < 1:
        parser.error("Layer counts must be nonnegative and --n-harm positive")
    return args


def resolve_lc_paths(spec):
    """Light-curve parquet files from a directory (recursive) or a glob pattern."""
    path = Path(spec)
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
    else:
        files = [Path(p) for p in sorted(glob.glob(spec, recursive=True))]
    if not files:
        raise FileNotFoundError(f"No light-curve files matched {spec!r}")
    return [str(p) for p in files]


def hash_fraction(object_id):
    """Map an object id to a fixed number in [0, 1), used for the train/validation split."""
    digest = hashlib.blake2b(
        str(object_id).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def split_indices(ids, limit, val_frac):
    """Objects with hash_fraction < val_frac go to validation; each split gets at least one."""
    n = len(ids) if limit is None else min(len(ids), limit)
    if n < 2:
        raise ValueError("At least two objects are required")
    hashes = [hash_fraction(ids[i]) for i in range(n)]
    val = [i for i, value in enumerate(hashes) if value < val_frac]
    train = [i for i, value in enumerate(hashes) if value >= val_frac]
    if not val:
        index = min(range(n), key=hashes.__getitem__)
        train.remove(index)
        val.append(index)
    if not train:
        index = max(range(n), key=hashes.__getitem__)
        val.remove(index)
        train.append(index)
    return train, val


class ExactDistributedSampler(Sampler):
    """DistributedSampler ordering without padding or duplicate indices."""

    def __init__(
        self, dataset, num_replicas=1, rank=0, shuffle=False, seed=0
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        count = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator().manual_seed(
                self.seed + self.epoch
            )
            indices = torch.randperm(
                count, generator=generator
            ).tolist()
        else:
            indices = list(range(count))
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self):
        count = len(self.dataset)
        return len(range(self.rank, count, self.num_replicas))


class EqualStepBatchSampler(Sampler):
    """Pad only empty final rank batches; real sampler indices stay disjoint."""

    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size
        self.total_batches = math.ceil(
            len(sampler.dataset) / (sampler.num_replicas * batch_size)
        )
        self.dummy_batches = set()

    def __iter__(self):
        indices = list(self.sampler)
        self.dummy_batches = set()
        for batch_number in range(self.total_batches):
            start = batch_number * self.batch_size
            batch = indices[start:start + self.batch_size]
            if not batch:
                self.dummy_batches.add(batch_number)
                batch = [0]
            yield batch

    def __len__(self):
        return self.total_batches


class _IndexedSubset(Subset):
    pass


def make_loader(dataset, *, batch_size, workers, device, shuffle=False,
                generator=None, sampler=None, batch_sampler=None):
    """DataLoader with the padding collate function and prefetching."""
    options = {
        "num_workers": workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = 4
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **options)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        sampler=sampler,
        **options,
    )


def learning_rate(step, total_steps, warmup, base_lr):
    """Linear warm-up for `warmup` steps, then cosine decay to zero at total_steps."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    decay_steps = max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup + 1) / decay_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _seed_from_parts(*values):
    """Deterministic 63-bit seed from any values (hash of their string form)."""
    payload = ":".join(str(value) for value in values).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


def attach_train_mask(cpu_batch, seed, step, rank, device, world_size=1):
    """Draw a fresh night mask for a training batch (seeded by seed, step and rank)."""
    if world_size == 1:
        mask_seed = (seed * 1_000_003 + step) % (2**63 - 1)
    else:
        mask_seed = _seed_from_parts(seed, step, rank)
    hidden = night_mask(
        cpu_batch["t"],
        cpu_batch["point_mask"],
        torch.Generator().manual_seed(mask_seed),
    )
    batch = {
        key: value.to(device, non_blocking=True)
        for key, value in cpu_batch.items()
    }
    batch["hidden"] = hidden.to(device, non_blocking=True)
    return batch


def attach_validation_mask(cpu_batch, seed, device):
    """Fixed night mask per object (seeded by object id), identical in every validation."""
    hidden_rows = []
    for row in range(cpu_batch["id"].numel()):
        object_id = int(cpu_batch["id"][row])
        generator = torch.Generator().manual_seed(
            _seed_from_parts(seed, object_id)
        )
        hidden_rows.append(
            night_mask(
                cpu_batch["t"][row:row + 1],
                cpu_batch["point_mask"][row:row + 1],
                generator,
            )
        )
    hidden = torch.cat(hidden_rows, dim=0)
    batch = {
        key: value.to(device, non_blocking=True)
        for key, value in cpu_batch.items()
    }
    batch["hidden"] = hidden.to(device, non_blocking=True)
    return batch


def metric_values(output):
    return {
        key: float(output[key].detach().float().item())
        for key in METRIC_KEYS
    }


def append_log(path, record):
    """Append one metrics record to log.jsonl and print a one-line summary."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
    print(
        f"{record['split']} epoch={record['epoch']} step={record['step']} "
        f"loss={record['loss']:.4f} fold={record['loss_fold']:.4f} "
        f"unf={record['loss_unf']:.4f} "
        f"fvu={record['loss_fvu']:.4f} rank={record['loss_rank']:.4f} "
        f"fvu_top={record['fvu_top_median']:.3f} "
        f"obj/s={record['objects_per_sec']:.1f}",
        flush=True,
    )


def _raw_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _restore_rng_state(state):
    # checkpoints are loaded with map_location=device, which also moves these tensors;
    # the RNG setters require CPU uint8 tensors
    torch.set_rng_state(torch.as_tensor(state["torch"]).to("cpu", torch.uint8))
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(torch.as_tensor(state["cuda"]).to("cpu", torch.uint8))


def _all_gather_object(value, world_size):
    gathered = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, value)
    else:
        gathered[0] = value
    return gathered


# Checkpoints are written to a temporary file and renamed, so a crash never leaves a
# half-written checkpoint.
def save_checkpoint(
    path, model, optimizer, args, epoch, batch_in_epoch,
    global_step, best_val, rank, world_size
):
    rng_states = _all_gather_object(_rng_state(), world_size)
    if rank == 0:
        state = {
            "model": _raw_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "best_val": best_val,
            "rng_states": rng_states,
            "schedule_total_steps": args.epochs * math.ceil(
                len(args._train_indices) / (args.batch * world_size)
            ),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
    if world_size > 1:
        dist.barrier()


def validate(model, loader, device, amp, seed, rank=0, world_size=1):
    """Evaluate each object alone so results do not depend on rank batching."""
    raw_model = _raw_model(model)
    raw_model.eval()
    started = time.perf_counter()
    local_rows = []
    local_sums = torch.zeros(
        len(METRIC_KEYS) + 1, dtype=torch.float64, device=device
    )

    with torch.no_grad():
        for cpu_batch in loader:
            for row in range(cpu_batch["id"].numel()):
                one = {
                    key: value[row:row + 1]
                    for key, value in cpu_batch.items()
                }
                batch = attach_validation_mask(one, seed, device)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=amp,
                ):
                    output = raw_model(batch)
                values = metric_values(output)
                object_id = int(one["id"][0])
                eligible_fvu = (
                    int(batch["hidden"].sum()) >= 3
                    and bool(batch["period_mask"].any())
                )
                local_rows.append((object_id, values, eligible_fvu))
                local_sums[:-1] += torch.tensor(
                    [values[key] for key in METRIC_KEYS],
                    dtype=torch.float64,
                    device=device,
                )
                local_sums[-1] += 1

    if world_size > 1:
        dist.all_reduce(local_sums, op=dist.ReduceOp.SUM)
    gathered = _all_gather_object(local_rows, world_size)
    elapsed = time.perf_counter() - started
    elapsed_tensor = torch.tensor(
        elapsed, dtype=torch.float64, device=device
    )
    if world_size > 1:
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)

    result = None
    if rank == 0:
        rows = sorted(
            (row for rank_rows in gathered for row in rank_rows),
            key=lambda row: row[0],
        )
        if len(rows) != int(local_sums[-1].item()):
            raise RuntimeError("Validation aggregation lost or duplicated objects")
        result = {}
        for key in METRIC_KEYS:
            if key in MEDIAN_KEYS:
                eligible = [
                    values[key]
                    for _, values, include in rows if include
                ]
                result[key] = (
                    statistics.median(eligible) if eligible else 0.0
                )
            else:
                result[key] = (
                    math.fsum(values[key] for _, values, _ in rows)
                    / max(1, len(rows))
                )
        result["objects_per_sec"] = (
            len(rows) / max(elapsed_tensor.item(), 1e-9)
        )

    if world_size > 1:
        payload = [result]
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
    return result


def _setup_distributed():
    """Read torchrun's environment variables and start the process group if needed."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            # Bind the process group to this process's GPU explicitly. Without it NCCL guesses
            # the device from the GLOBAL rank, which fails ('invalid device ordinal') when
            # 1-GPU tasks are spread over nodes and each sees only cuda:0 (job 17643035).
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                device_id=torch.device("cuda", local_rank),
            )
        else:
            dist.init_process_group(backend=backend, init_method="env://")
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    return rank, world_size, device


def main(argv=None):
    """Build the data, model and optimizer, then train, validate and checkpoint each epoch."""
    args = parse_args(argv)
    rank, world_size, device = _setup_distributed()
    try:
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True

        out_dir = Path(args.out)
        if rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            if device.type == "cuda":
                dist.barrier(device_ids=[device.index])
            else:
                dist.barrier()
        log_path = out_dir / "log.jsonl"

        if args.cache:
            train_dataset = CachedLightCurveDataset(
                args.cache,
                max_points=args.max_points,
                max_periods=args.max_periods,
                train=True,
                seed=args.seed,
            )
            val_dataset = CachedLightCurveDataset(
                args.cache,
                max_points=args.max_points,
                max_periods=args.max_periods,
                train=False,
                seed=args.seed,
            )
            ids = train_dataset.ids
        else:
            lc_paths = resolve_lc_paths(args.lc)
            mag_transform = (
                None if args.mag_transform == "none" else "asinh"
            )
            train_dataset = LightCurveDataset(
                lc_paths,
                args.cands,
                max_points=args.max_points,
                max_periods=args.max_periods,
                train=True,
                seed=args.seed,
                mag_transform=mag_transform,
            )
            val_dataset = LightCurveDataset(
                lc_paths,
                args.cands,
                max_points=args.max_points,
                max_periods=args.max_periods,
                train=False,
                seed=args.seed,
                mag_transform=mag_transform,
            )
            ids = [obj["id"] for obj in train_dataset.objects]

        train_indices, val_indices = split_indices(
            ids, args.limit, args.val_frac
        )
        args._train_indices = train_indices
        train_subset = _IndexedSubset(train_dataset, train_indices)
        val_subset = _IndexedSubset(val_dataset, val_indices)

        model = PretrainModel(
            encoder=args.encoder,
            d_model=args.d_model,
            likelihood=args.likelihood,
            fvu_weight=args.fvu_weight,
            rank_weight=args.rank_weight,
            rank_tau=args.rank_tau,
            fvu_cap=args.fvu_cap,
            fold_kwargs={
                "n_heads": args.n_heads,
                "n_layers": args.fold_layers,
                "n_cand_layers": args.cand_layers,
                "n_harm": args.n_harm,
                "fold_chunk": args.fold_chunk,
                "grad_checkpoint": args.grad_checkpoint,
            },
            unfolded_kwargs={
                "n_heads": args.n_heads,
                "n_layers": args.unf_layers,
                "grad_checkpoint": args.grad_checkpoint,
            },
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        batches_per_epoch = math.ceil(
            len(train_subset) / (args.batch * world_size)
        )
        schedule_total_steps = args.epochs * batches_per_epoch
        start_epoch = 0
        start_batch = 0
        global_step = 0
        best_val = math.inf
        if args.resume:
            state = torch.load(
                args.resume, map_location=device, weights_only=False
            )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"])
            start_batch = int(state.get("batch_in_epoch", 0))
            global_step = int(state["global_step"])
            best_val = float(state.get("best_val", math.inf))
            saved_states = state.get("rng_states")
            if saved_states and rank < len(saved_states):
                _restore_rng_state(saved_states[rank])

        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )

        val_sampler = ExactDistributedSampler(
            val_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        val_loader = make_loader(
            val_subset,
            batch_size=args.batch,
            workers=args.workers,
            device=device,
            sampler=val_sampler,
        )

        run_start = time.perf_counter()
        run_objects = 0
        stopped = False
        for epoch in range(start_epoch, args.epochs):
            train_dataset.set_epoch(epoch)
            train_sampler = ExactDistributedSampler(
                train_subset,
                num_replicas=world_size,
                rank=rank,
                shuffle=world_size > 1,
                seed=args.seed,
            )
            train_sampler.set_epoch(epoch)
            batch_sampler = None
            if world_size > 1:
                batch_sampler = EqualStepBatchSampler(
                    train_sampler, args.batch
                )
                train_loader = make_loader(
                    train_subset,
                    batch_size=args.batch,
                    workers=args.workers,
                    device=device,
                    batch_sampler=batch_sampler,
                )
            else:
                train_loader = make_loader(
                    train_subset,
                    batch_size=args.batch,
                    workers=args.workers,
                    device=device,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(
                        args.seed + epoch
                    ),
                )

            model.train()
            epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            interval_start = time.perf_counter()
            interval_objects = 0

            for batch_number, cpu_batch in enumerate(train_loader):
                if epoch == start_epoch and batch_number < start_batch:
                    continue
                dummy = (
                    batch_sampler is not None
                    and batch_number in batch_sampler.dummy_batches
                )
                batch = attach_train_mask(
                    cpu_batch,
                    args.seed,
                    global_step,
                    rank,
                    device,
                    world_size,
                )
                local_count = 0 if dummy else batch["t"].shape[0]
                count_tensor = torch.tensor(
                    float(local_count), device=device
                )
                if world_size > 1:
                    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                global_count = int(count_tensor.item())

                lr = learning_rate(
                    global_step,
                    schedule_total_steps,
                    args.warmup,
                    args.lr,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    output = model(batch)
                    scale = (
                        world_size * local_count / max(global_count, 1)
                    )
                    loss = output["loss"] * scale
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

                global_step += 1
                run_objects += global_count
                interval_objects += global_count
                if global_step % args.log_every == 0:
                    values = metric_values(output)
                    sums = torch.tensor(
                        [values[key] * local_count for key in METRIC_KEYS],
                        dtype=torch.float64,
                        device=device,
                    )
                    if world_size > 1:
                        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                    elapsed = torch.tensor(
                        time.perf_counter() - interval_start,
                        dtype=torch.float64,
                        device=device,
                    )
                    if world_size > 1:
                        dist.all_reduce(
                            elapsed, op=dist.ReduceOp.MAX
                        )
                    if rank == 0:
                        record = {
                            "split": "train",
                            "step": global_step,
                            "epoch": epoch,
                            "lr": lr,
                            "global_batch": args.batch * world_size,
                            **{
                                key: float(sums[i].item())
                                / max(global_count, 1)
                                for i, key in enumerate(METRIC_KEYS)
                            },
                            "objects_per_sec": interval_objects
                            / max(elapsed.item(), 1e-9),
                        }
                        append_log(log_path, record)
                    interval_start = time.perf_counter()
                    interval_objects = 0

                if (
                    args.max_steps is not None
                    and global_step >= args.max_steps
                ):
                    next_epoch = epoch
                    next_batch = batch_number + 1
                    if next_batch == len(train_loader):
                        next_epoch += 1
                        next_batch = 0
                    save_checkpoint(
                        out_dir / "last.pt",
                        model,
                        optimizer,
                        args,
                        next_epoch,
                        next_batch,
                        global_step,
                        best_val,
                        rank,
                        world_size,
                    )
                    stopped = True
                    break

            if stopped:
                break

            val_metrics = validate(
                model,
                val_loader,
                device,
                args.amp and device.type == "cuda",
                args.seed,
                rank,
                world_size,
            )
            peak = (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0
            )
            peaks = _all_gather_object(peak, world_size)
            epoch_elapsed = torch.tensor(
                time.perf_counter() - epoch_start,
                dtype=torch.float64,
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(
                    epoch_elapsed, op=dist.ReduceOp.MAX
                )
            if rank == 0:
                record = {
                    "split": "val",
                    "step": global_step,
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "global_batch": args.batch * world_size,
                    "gpu_peak_bytes_per_rank": peaks,
                    **val_metrics,
                }
                append_log(log_path, record)

            improved = val_metrics["loss"] < best_val
            if improved:
                best_val = val_metrics["loss"]
            save_checkpoint(
                out_dir / "last.pt",
                model,
                optimizer,
                args,
                epoch + 1,
                0,
                global_step,
                best_val,
                rank,
                world_size,
            )
            if improved:
                save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    optimizer,
                    args,
                    epoch + 1,
                    0,
                    global_step,
                    best_val,
                    rank,
                    world_size,
                )
            start_batch = 0

        elapsed = torch.tensor(
            time.perf_counter() - run_start,
            dtype=torch.float64,
            device=device,
        )
        if world_size > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if rank == 0:
            summary = {
                "split": "final",
                "step": global_step,
                "steps": global_step,
                "epoch": min(args.epochs - 1, epoch),
                "elapsed_seconds": elapsed.item(),
                "objects_per_sec": run_objects / max(
                    elapsed.item(), 1e-9
                ),
                "global_batch": args.batch * world_size,
                "lr": optimizer.param_groups[0]["lr"],
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, allow_nan=False) + "\n")
            print(
                f"final steps={global_step} "
                f"elapsed={elapsed.item():.1f}s "
                f"obj/s={summary['objects_per_sec']:.1f}",
                flush=True,
            )
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
