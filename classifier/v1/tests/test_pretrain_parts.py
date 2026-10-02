# test_pretrain_parts.py
import pytest
import torch

from fold_branch import FoldBranch
from masking import night_mask
from unfolded_branch import UnfoldedBranch


def make_batch(device):
    t = torch.tensor(
        [
            [0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1, 0.0, 0.0],
            [0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1, 5.0, 5.1],
        ],
        device=device,
    )
    point_mask = torch.tensor(
        [[True] * 10 + [False] * 2, [True] * 12],
        device=device,
    )
    return {
        "t": t,
        "band": torch.tensor(
            [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
             [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]],
            device=device,
        ),
        "mag": torch.linspace(-1, 1, 24, device=device).reshape(2, 12),
        "point_mask": point_mask,
        "periods": torch.tensor(
            [[1.0, 2.0, 1.0], [1.5, 2.5, 3.0]], device=device
        ),
        "cycles": torch.tensor(
            [[5.0, 2.5, 1.0], [4.0, 2.4, 2.0]], device=device
        ),
        "period_mask": torch.tensor(
            [[True, True, False], [True, True, True]], device=device
        ),
        "baseline": torch.tensor([4.1, 5.1], device=device),
        "id": torch.tensor([11, 12], device=device),
    }


def with_hidden(batch):
    result = dict(batch)
    hidden = torch.zeros_like(batch["point_mask"])
    hidden[:, 2:4] = True
    hidden[:, 6:8] = True
    result["hidden"] = hidden
    return result


def fold_model(encoder, device, *, checkpoint=False):
    kwargs = {
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "n_cand_layers": 1,
        "dropout": 0.0,
        "fold_chunk": 2,
        "grad_checkpoint": checkpoint,
        "encoder": encoder,
        "point_head": True,
    }
    if encoder == "cnn":
        kwargs["cnn_kwargs"] = {
            "n_bins": 16,
            "widths": (8, 16),
            "blocks_per_stage": 1,
        }
    return FoldBranch(**kwargs).to(device)


def test_night_mask():
    t = torch.tensor(
        [
            [0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1, 0.0, 0.0],
            [0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1, 5.0, 5.1],
        ]
    )
    point_mask = torch.tensor(
        [[True] * 10 + [False] * 2, [True] * 12]
    )
    kwargs = dict(
        frac=0.30,
        min_visible_nights=3,
        min_visible_points=6,
    )
    hidden = night_mask(
        t, point_mask, torch.Generator().manual_seed(123), **kwargs
    )
    repeated = night_mask(
        t, point_mask, torch.Generator().manual_seed(123), **kwargs
    )
    assert torch.equal(hidden, repeated)
    assert not (hidden & ~point_mask).any()

    for row, n_nights in enumerate((5, 6)):
        for night in range(n_nights):
            members = hidden[row, 2 * night:2 * night + 2]
            assert bool(members.all()) or bool((~members).all())
        assert int(hidden[row].sum()) == 2 * round(0.30 * n_nights)
        assert int((point_mask[row] & ~hidden[row]).sum()) >= 6
        assert n_nights - int(hidden[row].sum()) // 2 >= 3

    constrained = night_mask(
        t, point_mask, torch.Generator().manual_seed(123),
        frac=0.8, min_visible_nights=4, min_visible_points=9,
    )
    assert not constrained[0].any()
    assert int(constrained[1].sum()) <= 2
    assert not (constrained & ~point_mask).any()


@pytest.mark.parametrize("encoder", ["rope", "cnn"])
def test_fold_hidden_magnitude_cannot_leak(encoder):
    torch.manual_seed(7)
    model = fold_model(encoder, "cpu").eval()
    batch = with_hidden(make_batch("cpu"))
    changed = dict(batch)
    changed["mag"] = batch["mag"].clone()
    changed["mag"][batch["hidden"]] += 5.0

    with torch.no_grad():
        before = model(batch)
        after = model(changed)

    for key in ("z", "cand_scores", "cand_vecs", "fold_pred"):
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)


def test_unfolded_hidden_magnitude_cannot_leak():
    torch.manual_seed(7)
    model = UnfoldedBranch(
        d_model=16, n_heads=2, n_layers=1, dropout=0.0
    ).eval()
    batch = with_hidden(make_batch("cpu"))
    changed = dict(batch)
    changed["mag"] = batch["mag"].clone()
    changed["mag"][batch["hidden"]] -= 5.0

    with torch.no_grad():
        before = model(batch)
        after = model(changed)

    for key in ("z", "tokens", "pool_weights"):
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    assert not before["pool_weights"][batch["hidden"]].any()


@pytest.mark.parametrize("encoder", ["rope", "cnn"])
def test_absent_hidden_matches_all_false(encoder):
    torch.manual_seed(11)
    batch = make_batch("cpu")
    all_visible = dict(batch, hidden=torch.zeros_like(batch["point_mask"]))
    fold = fold_model(encoder, "cpu").eval()
    unfolded = UnfoldedBranch(
        d_model=16, n_heads=2, n_layers=1, dropout=0.0
    ).eval()

    with torch.no_grad():
        old_fold = fold(batch)
        visible_fold = fold(all_visible)
        old_unfolded = unfolded(batch)
        visible_unfolded = unfolded(all_visible)

    for key in ("z", "cand_scores", "cand_vecs"):
        torch.testing.assert_close(
            old_fold[key], visible_fold[key], atol=0, rtol=0
        )
    for key in ("z", "tokens", "pool_weights"):
        torch.testing.assert_close(
            old_unfolded[key], visible_unfolded[key], atol=0, rtol=0
        )


@pytest.mark.parametrize("encoder", ["rope", "cnn"])
@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_fold_predictions_and_backward(encoder, device):
    torch.manual_seed(13)
    batch = with_hidden(make_batch(device))
    model = fold_model(encoder, device, checkpoint=True).train()

    output = model(batch)
    pred = output["fold_pred"]
    assert pred.shape == (2, 3, 12)
    assert torch.isfinite(pred).all()
    assert torch.equal(
        pred.masked_select(~batch["point_mask"][:, None, :].expand_as(pred)),
        torch.zeros_like(
            pred.masked_select(
                ~batch["point_mask"][:, None, :].expand_as(pred)
            )
        ),
    )
    assert torch.equal(pred[:, 2][0], torch.zeros_like(pred[:, 2][0]))

    hidden_candidates = (
        batch["hidden"][:, None, :]
        & batch["period_mask"][:, :, None]
    )
    loss = pred[hidden_candidates].square().mean() + output["z"].square().mean()
    loss.backward()
    assert model.point_head.weight.grad is not None
    assert torch.isfinite(model.point_head.weight.grad).all()
