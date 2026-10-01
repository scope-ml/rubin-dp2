# test_fvu.py
import math

import pytest
import torch
from torch import nn

from pretrain import PretrainModel
from train_pretrain import METRIC_KEYS, parse_args


def test_fvu_perfect_and_mean_predictions():
    target = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
    pred = torch.tensor(
        [[[-1.0, 0.0, 1.0, 2.0], [0.5, 0.5, 0.5, 0.5]]]
    )
    hidden = torch.ones(1, 4, dtype=torch.bool)
    valid = torch.ones(1, 2, dtype=torch.bool)
    scores = torch.tensor([[2.0, -2.0]])

    loss_fvu, loss_rank, diagnostics = PretrainModel.fold_quality(
        pred, scores, target, hidden, valid
    )
    assert loss_fvu.item() == pytest.approx(0.5)
    assert torch.isfinite(loss_rank)
    assert diagnostics["fvu_top_median"].item() == pytest.approx(0.0)
    assert diagnostics["fvu_best_median"].item() == pytest.approx(0.0)
    assert diagnostics["frac_top_fvu_lt1"].item() == 1.0
    assert diagnostics["frac_top_fvu_lt03"].item() == 1.0


def test_constant_target_uses_denominator_floor():
    target = torch.full((1, 4), 2.0)
    pred = torch.full((1, 1, 4), 3.0)
    hidden = torch.ones(1, 4, dtype=torch.bool)
    valid = torch.ones(1, 1, dtype=torch.bool)
    scores = torch.zeros(1, 1)

    loss_fvu, _, diagnostics = PretrainModel.fold_quality(
        pred, scores, target, hidden, valid, fvu_cap=200.0
    )
    # Numerator = 4, denominator = 4 * 0.01.
    assert loss_fvu.item() == pytest.approx(100.0)
    assert diagnostics["fvu_top_median"].item() == pytest.approx(100.0)


def test_rank_loss_prefers_the_lower_fvu_candidate():
    target = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
    pred = torch.tensor(
        [[[-1.0, 0.0, 1.0, 2.0], [0.5, 0.5, 0.5, 0.5]]]
    )
    hidden = torch.ones(1, 4, dtype=torch.bool)
    valid = torch.ones(1, 2, dtype=torch.bool)

    _, good_rank, _ = PretrainModel.fold_quality(
        pred, torch.tensor([[3.0, -3.0]]),
        target, hidden, valid,
    )
    _, bad_rank, _ = PretrainModel.fold_quality(
        pred, torch.tensor([[-3.0, 3.0]]),
        target, hidden, valid,
    )
    assert good_rank < bad_rank


def test_padding_and_objects_with_fewer_than_three_hidden_points_are_ignored():
    target = torch.tensor(
        [[-1.0, 0.0, 1.0, 2.0],
         [5.0, 5.0, 5.0, 5.0]]
    )
    pred = torch.tensor(
        [
            [[-1.0, 0.0, 1.0, 2.0],
             [0.5, 0.5, 0.5, 0.5],
             [1e4, 1e4, 1e4, 1e4]],
            [[1e4, 1e4, 1e4, 1e4],
             [1e4, 1e4, 1e4, 1e4],
             [1e4, 1e4, 1e4, 1e4]],
        ]
    )
    scores = torch.tensor(
        [[2.0, -2.0, -torch.inf],
         [2.0, -2.0, -torch.inf]]
    )
    hidden = torch.tensor(
        [[True, True, True, True],
         [True, True, False, False]]
    )
    valid = torch.tensor(
        [[True, True, False],
         [True, True, False]]
    )

    combined = PretrainModel.fold_quality(
        pred, scores, target, hidden, valid
    )
    first_only = PretrainModel.fold_quality(
        pred[:1], scores[:1], target[:1],
        hidden[:1], valid[:1],
    )
    torch.testing.assert_close(combined[0], first_only[0])
    torch.testing.assert_close(combined[1], first_only[1])
    for key in combined[2]:
        torch.testing.assert_close(
            combined[2][key], first_only[2][key]
        )

    pred[:, 2] = -1e9
    changed = PretrainModel.fold_quality(
        pred, scores, target, hidden, valid
    )
    torch.testing.assert_close(changed[0], combined[0])
    torch.testing.assert_close(changed[1], combined[1])


class _FixedFold(nn.Module):
    def __init__(self, predictions, scores):
        super().__init__()
        self.register_buffer("predictions", predictions)
        self.register_buffer("scores", scores)

    def forward(self, batch):
        batch_size, n_candidates, _ = self.predictions.shape
        return {
            "fold_pred": self.predictions,
            "cand_scores": self.scores,
            "cand_vecs": self.predictions.new_zeros(
                batch_size, n_candidates, 1
            ),
            "z": self.predictions.new_zeros(batch_size, 1),
        }


class _FixedUnfolded(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.register_buffer("tokens", tokens)

    def forward(self, batch):
        return {
            "tokens": self.tokens,
            "z": self.tokens.new_zeros(self.tokens.shape[0], 1),
            "pool_weights": self.tokens.new_zeros(self.tokens.shape[:2]),
        }


@pytest.mark.parametrize("likelihood", ["laplace", "gaussian"])
def test_zero_weights_preserve_previous_total_exactly(likelihood):
    model = PretrainModel(
        d_model=16,
        likelihood=likelihood,
        fvu_weight=0.0,
        rank_weight=0.0,
        fold_kwargs={"n_layers": 0, "n_cand_layers": 0},
        unfolded_kwargs={"n_layers": 0},
    )
    target = torch.tensor([[0.0, 1.0, -1.0, 0.5]])
    predictions = torch.tensor(
        [[[0.0, 0.9, -0.8, 0.5],
          [1.0, 1.2, -0.4, 0.0]]]
    )
    scores = torch.tensor([[1.0, -1.0]])
    model.fold = _FixedFold(predictions, scores)
    model.unfolded = _FixedUnfolded(torch.zeros(1, 4, 16))
    with torch.no_grad():
        model.unfolded_point_head.weight.zero_()
        model.unfolded_point_head.bias.zero_()

    hidden = torch.tensor([[True, True, True, False]])
    batch = {
        "mag": target,
        "hidden": hidden,
        "point_mask": torch.ones_like(hidden),
        "period_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    output = model(batch)

    log_scale = (
        model.log_b if likelihood == "laplace" else model.log_sigma
    )
    old_fold, _ = PretrainModel.fold_mixture(
        predictions,
        scores,
        target,
        hidden,
        batch["period_mask"],
        log_scale,
        likelihood=likelihood,
    )
    residual = target[hidden]
    old_unfolded = (
        residual.abs().mean()
        if likelihood == "laplace"
        else residual.square().mean()
    )
    assert torch.equal(output["loss"], old_fold + old_unfolded)
    assert torch.isfinite(output["loss_fvu"])
    assert torch.isfinite(output["loss_rank"])


def test_cli_flags_and_logged_metric_names():
    args = parse_args(
        [
            "--lc", "lc.parquet",
            "--cands", "cands.parquet",
            "--out", "run",
            "--fvu-weight", "0.4",
            "--rank-weight", "0.2",
            "--rank-tau", "0.3",
            "--fvu-cap", "4.0",
        ]
    )
    assert args.fvu_weight == pytest.approx(0.4)
    assert args.rank_weight == pytest.approx(0.2)
    assert args.rank_tau == pytest.approx(0.3)
    assert args.fvu_cap == pytest.approx(4.0)
    assert {
        "loss_fvu",
        "loss_rank",
        "fvu_top_median",
        "fvu_best_median",
        "frac_top_fvu_lt1",
        "frac_top_fvu_lt03",
    }.issubset(METRIC_KEYS)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
def test_fvu_losses_finite_under_cuda_bf16_autocast():
    device = "cuda"
    torch.manual_seed(7)
    model = PretrainModel(
        d_model=16,
        fvu_weight=0.5,
        rank_weight=0.2,
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
    assert torch.isfinite(model.fold.point_head.weight.grad).all()
