# test_pretrain.py
import json
import math

import pytest
import torch

from pretrain import PretrainModel
from train_pretrain import main


def random_batch(device="cpu", batch_size=3, n_points=24, n_candidates=4):
    torch.manual_seed(21)
    t = torch.arange(n_points, device=device).float()[None, :]
    t = t.expand(batch_size, -1).clone()
    t += torch.arange(batch_size, device=device)[:, None] * 0.03
    point_mask = torch.ones(
        batch_size, n_points, dtype=torch.bool, device=device
    )
    point_mask[0, -3:] = False
    hidden = torch.zeros_like(point_mask)
    hidden[:, 3:7] = True
    hidden[:, 12:15] = True
    period_mask = torch.ones(
        batch_size, n_candidates, dtype=torch.bool, device=device
    )
    period_mask[0, -1] = False
    return {
        "t": t,
        "band": torch.randint(
            0, 6, (batch_size, n_points), device=device
        ),
        "mag": torch.randn(batch_size, n_points, device=device),
        "point_mask": point_mask,
        "hidden": hidden,
        "periods": torch.rand(
            batch_size, n_candidates, device=device
        ) * 2 + 0.5,
        "cycles": torch.rand(
            batch_size, n_candidates, device=device
        ) * 10 + 1,
        "period_mask": period_mask,
        "baseline": torch.full((batch_size,), float(n_points), device=device),
        "id": torch.arange(batch_size, device=device),
    }


@pytest.mark.parametrize("encoder", ["rope", "cnn"])
def test_forward_backward(encoder):
    kwargs = {
        "d_model": 16,
        "fold_kwargs": {
            "n_layers": 1,
            "n_cand_layers": 1,
            "dropout": 0.0,
            "fold_chunk": 3,
        },
        "unfolded_kwargs": {
            "n_layers": 1,
            "dropout": 0.0,
        },
    }
    if encoder == "cnn":
        kwargs["fold_kwargs"]["cnn_kwargs"] = {
            "n_bins": 16,
            "widths": (8, 16),
            "blocks_per_stage": 1,
        }
    model = PretrainModel(encoder=encoder, **kwargs)
    output = model(random_batch())
    for key in (
        "loss",
        "loss_fold",
        "loss_unf",
        "sigma",
        "cand_entropy",
        "top_is_best",
        "mse_best_fold",
        "mse_top_fold",
    ):
        assert torch.isfinite(output[key])
    assert output["cand_entropy"] >= 0
    assert 0 <= output["top_is_best"] <= 1
    output["loss"].backward()
    assert model.unfolded_point_head.weight.grad is not None
    assert model.fold.point_head.weight.grad is not None


def test_hand_checked_mixture_prefers_accurate_candidate():
    target = torch.tensor([[0.2, -0.4]])
    hidden = torch.tensor([[True, True]])
    period_mask = torch.tensor([[True, True]])
    fold_pred = torch.tensor(
        [[[0.2, -0.4], [1.2, 0.6]]]
    )
    log_sigma = torch.tensor(math.log(0.5))
    favour_good = torch.tensor([[3.0, -3.0]])
    favour_bad = torch.tensor([[-3.0, 3.0]])

    good_loss, _ = PretrainModel.fold_mixture(
        fold_pred, favour_good, target, hidden, period_mask, log_sigma
    )
    bad_loss, _ = PretrainModel.fold_mixture(
        fold_pred, favour_bad, target, hidden, period_mask, log_sigma
    )
    assert good_loss < bad_loss


@pytest.mark.slow
def test_rope_learns_true_period_on_synthetic_nights():
    torch.manual_seed(31)
    generator = torch.Generator().manual_seed(31)
    n_objects = 64
    n_nights = 40
    n_points = 2 * n_nights
    n_candidates = 12

    nights = torch.arange(n_nights).repeat_interleave(2).float()
    within_night = torch.tensor(
        [0.0, 1.0 / 48.0]
    ).repeat(n_nights)
    t = (nights + within_night)[None, :].repeat(n_objects, 1)
    true_period = 0.3 + 2.7 * torch.rand(
        n_objects, generator=generator
    )
    phase_offset = 2 * math.pi * torch.rand(
        n_objects, generator=generator
    )
    mag = 0.8 * torch.sin(
        2 * math.pi * t / true_period[:, None]
        + phase_offset[:, None]
    )
    mag += 0.02 * torch.randn(
        n_objects, n_points, generator=generator
    )

    candidates = torch.empty(n_objects, n_candidates)
    true_index = torch.randint(
        n_candidates, (n_objects,), generator=generator
    )
    for row in range(n_objects):
        decoys = []
        while len(decoys) < n_candidates - 1:
            proposal = 0.1 + 9.9 * torch.rand(
                (), generator=generator
            ).item()
            if all(
                abs(proposal / (factor * true_period[row].item()) - 1) > 0.05
                for factor in (0.5, 1.0, 2.0)
            ):
                decoys.append(proposal)
        values = decoys.copy()
        values.insert(int(true_index[row]), float(true_period[row]))
        candidates[row] = torch.tensor(values)

    batch = {
        "t": t,
        "band": torch.zeros(n_objects, n_points, dtype=torch.long),
        "mag": mag,
        "point_mask": torch.ones(n_objects, n_points, dtype=torch.bool),
        "periods": candidates,
        "cycles": 39.0 / candidates,
        "period_mask": torch.ones(
            n_objects, n_candidates, dtype=torch.bool
        ),
        "baseline": torch.full((n_objects,), 39.0),
        "id": torch.arange(n_objects),
    }
    model = PretrainModel(
        encoder="rope",
        d_model=32,
        fold_kwargs={
            "n_heads": 2,
            "n_layers": 1,
            "n_cand_layers": 1,
            "dropout": 0.0,
            "fold_chunk": 96,
        },
        unfolded_kwargs={
            "n_heads": 2,
            "n_layers": 1,
            "dropout": 0.0,
        },
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mask_seed = 430

    def masked_batch(indices):
        from masking import night_mask

        subset = {key: value[indices] for key, value in batch.items()}
        subset["hidden"] = night_mask(
            subset["t"],
            subset["point_mask"],
            torch.Generator().manual_seed(mask_seed),
        )
        return subset

    model.eval()
    with torch.no_grad():
        initial_loss = model(masked_batch(torch.arange(n_objects)))[
            "loss_fold"
        ].item()

    model.train()
    for step in range(240):
        indices = torch.randperm(
            n_objects, generator=generator
        )[:8]
        subset = masked_batch(indices)
        optimizer.zero_grad(set_to_none=True)
        loss = model(subset)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        evaluated = model(masked_batch(torch.arange(n_objects)))
        fold_output = model.fold(
            masked_batch(torch.arange(n_objects))
        )
    accuracy = (
        fold_output["cand_scores"].argmax(dim=-1) == true_index
    ).float().mean()
    assert accuracy >= 0.60
    assert evaluated["loss_fold"].item() < initial_loss


def test_train_cli_smoke(tmp_path, monkeypatch):
    """Exercise the CLI on a tiny parquet-backed dataset."""
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    lc_path = tmp_path / "lc.parquet"
    cands_path = tmp_path / "cands.parquet"
    rows = []
    candidate_rows = []
    for object_id in range(8):
        times = [
            float(night) + offset
            for night in range(12)
            for offset in (0.0, 1.0 / 48.0)
        ]
        rows.append(
            {
                "id": object_id,
                "t": times,
                "band": [0] * len(times),
                "mag": [
                    math.sin(2 * math.pi * value / 1.7)
                    for value in times
                ],
            }
        )
        candidate_rows.append(
            {
                "id": object_id,
                "periods": [1.7, 2.4],
                "cycles": [11.0 / 1.7, 11.0 / 2.4],
            }
        )
    parquet.write_table(pyarrow.Table.from_pylist(rows), lc_path)
    parquet.write_table(
        pyarrow.Table.from_pylist(candidate_rows), cands_path
    )

    # This adapter keeps the smoke test independent of the production
    # parquet column layout while still reading the files through parquet.
    class TinyParquetDataset:
        def __init__(
            self, lc_paths, cands_path, max_points=512,
            max_periods=256, train=True, seed=0, **kwargs
        ):
            self.rows = parquet.read_table(lc_paths[0]).to_pylist()
            self.candidates = {
                row["id"]: row
                for row in parquet.read_table(cands_path).to_pylist()
            }
            self.objects = [{"id": row["id"]} for row in self.rows]

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            cand = self.candidates[row["id"]]
            return {
                "t": torch.tensor(row["t"], dtype=torch.float32),
                "band": torch.tensor(row["band"], dtype=torch.long),
                "mag": torch.tensor(row["mag"], dtype=torch.float32),
                "point_mask": torch.ones(len(row["t"]), dtype=torch.bool),
                "periods": torch.tensor(
                    cand["periods"], dtype=torch.float32
                ),
                "cycles": torch.tensor(
                    cand["cycles"], dtype=torch.float32
                ),
                "period_mask": torch.ones(
                    len(cand["periods"]), dtype=torch.bool
                ),
                "baseline": torch.tensor(11.0),
                "id": torch.tensor(row["id"]),
            }

    def tiny_collate(items):
        return {
            key: torch.stack([item[key] for item in items])
            for key in items[0]
        }

    import train_pretrain

    monkeypatch.setattr(
        train_pretrain, "LightCurveDataset", TinyParquetDataset
    )
    monkeypatch.setattr(train_pretrain, "collate", tiny_collate)
    out_dir = tmp_path / "run"
    main(
        [
            "--lc", str(lc_path),
            "--cands", str(cands_path),
            "--out", str(out_dir),
            "--epochs", "1",
            "--batch", "2",
            "--workers", "0",
            "--limit", "6",
            "--log-every", "1",
            "--max-points", "32",
            "--max-periods", "2",
            "--fold-chunk", "2",
        ]
    )

    assert (out_dir / "log.jsonl").exists()
    assert (out_dir / "last.pt").exists()
    assert (out_dir / "best.pt").exists()
    records = [
        json.loads(line)
        for line in (out_dir / "log.jsonl").read_text().splitlines()
    ]
    assert any(record["split"] == "train" for record in records)
    assert any(record["split"] == "val" for record in records)
    for record in records:
        if record["split"] not in ("train", "val"):
            continue  # the end-of-run summary record carries timing, not losses
        for key in (
            "loss",
            "loss_fold",
            "loss_unf",
            "sigma",
            "cand_entropy",
            "top_is_best",
            "mse_best_fold",
            "mse_top_fold",
            "objects_per_sec",
        ):
            assert math.isfinite(record[key])
