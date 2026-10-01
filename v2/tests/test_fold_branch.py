# test_fold_branch.py
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fold_branch import FoldBranch, FoldEncoder


CNN_KWARGS = {
    "n_bins": 16,
    "widths": (8, 8),
    "blocks_per_stage": 1,
}


def make_batch(device="cpu", batch_size=2, n_points=50, n_candidates=7):
    torch.manual_seed(123)
    point_mask = torch.ones(
        batch_size, n_points, dtype=torch.bool, device=device
    )
    point_mask[0, -5:] = False
    period_mask = torch.ones(
        batch_size, n_candidates, dtype=torch.bool, device=device
    )
    period_mask[0, -2:] = False
    period_mask[1, -1:] = False

    t = torch.sort(
        torch.rand(batch_size, n_points, device=device) * 257, dim=-1
    ).values
    periods = 0.1 + 5 * torch.rand(
        batch_size, n_candidates, device=device
    )
    return {
        "t": t,
        "band": torch.randint(
            0, 6, (batch_size, n_points), device=device
        ),
        "mag": torch.randn(batch_size, n_points, device=device),
        "point_mask": point_mask,
        "periods": periods,
        "cycles": 257 / periods,
        "period_mask": period_mask,
        "baseline": torch.full((batch_size,), 257.0, device=device),
        "id": torch.arange(batch_size, device=device),
    }


def assert_close(a, b):
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@pytest.fixture
def fold_inputs():
    torch.manual_seed(41)
    tokens = torch.randn(3, 17, 32)
    phases = torch.rand(3, 17)
    mask = torch.ones(3, 17, dtype=torch.bool)
    return tokens, phases, mask


@pytest.mark.parametrize("shift", [0.13, 0.5, 0.97])
def test_phase_shift_invariance(fold_inputs, shift):
    encoder = FoldEncoder(
        d_model=32, n_heads=4, n_layers=3, dropout=0.1
    ).eval()
    tokens, phases, mask = fold_inputs
    with torch.no_grad():
        original = encoder(tokens, phases, mask)
        shifted = encoder(tokens, torch.remainder(phases + shift, 1), mask)
    assert_close(original, shifted)


def test_point_permutation_invariance(fold_inputs):
    encoder = FoldEncoder(
        d_model=32, n_heads=4, n_layers=3, dropout=0.1
    ).eval()
    tokens, phases, mask = fold_inputs
    permutation = torch.randperm(tokens.shape[1])
    with torch.no_grad():
        original = encoder(tokens, phases, mask)
        permuted = encoder(
            tokens[:, permutation],
            phases[:, permutation],
            mask[:, permutation],
        )
    assert_close(original, permuted)


def test_padding_invariance(fold_inputs):
    encoder = FoldEncoder(
        d_model=32, n_heads=4, n_layers=3, dropout=0.1
    ).eval()
    tokens, phases, mask = fold_inputs
    extra_tokens = torch.randn(3, 6, 32)
    extra_phases = torch.rand(3, 6)
    extra_mask = torch.zeros(3, 6, dtype=torch.bool)
    with torch.no_grad():
        original = encoder(tokens, phases, mask)
        padded = encoder(
            torch.cat((tokens, extra_tokens), dim=1),
            torch.cat((phases, extra_phases), dim=1),
            torch.cat((mask, extra_mask), dim=1),
        )
    assert_close(original, padded)


@pytest.mark.parametrize("encoder", ["rope", "cnn"])
def test_candidate_set_permutation_and_padding(encoder):
    batch = make_batch()
    model = FoldBranch(
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_cand_layers=2,
        fold_chunk=3,
        encoder=encoder,
        cnn_kwargs=CNN_KWARGS if encoder == "cnn" else None,
    ).eval()
    permutation = torch.tensor([4, 1, 6, 0, 3, 5, 2])
    changed = dict(batch)
    for key in ("periods", "cycles", "period_mask"):
        changed[key] = batch[key][:, permutation]

    with torch.no_grad():
        original = model(batch)
        permuted = model(changed)

    assert_close(original["z"], permuted["z"])
    assert_close(
        original["cand_vecs"][:, permutation], permuted["cand_vecs"]
    )
    finite = changed["period_mask"]
    assert_close(
        original["cand_scores"][:, permutation][finite],
        permuted["cand_scores"][finite],
    )
    assert torch.isneginf(permuted["cand_scores"][~finite]).all()


def test_phase_precision():
    t = torch.tensor([[257.0]], dtype=torch.float32)
    periods = torch.tensor([[0.0035]], dtype=torch.float32)
    actual = FoldBranch.compute_phase(t, periods)[0, 0, 0]
    reference = torch.remainder(
        t.double()[0, 0] / periods.double()[0, 0], 1.0
    )
    assert abs(actual.item() - reference.item()) < 1e-6


def test_fold_chunk_independence():
    batch = make_batch()
    model = FoldBranch(
        d_model=32, n_heads=4, n_layers=2, fold_chunk=3
    ).eval()
    with torch.no_grad():
        small = model(batch)
        model.fold_chunk = 1000
        large = model(batch)
    for key in small:
        finite = torch.isfinite(small[key])
        assert_close(small[key][finite], large[key][finite])
        assert torch.equal(
            torch.isneginf(small[key]), torch.isneginf(large[key])
        )


def test_attention_pool_can_focus_on_sparse_dip():
    torch.manual_seed(2026)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        n_folds, n_points = 16, 60
        phases = torch.linspace(0, 1, n_points + 1)[:-1]
        phases = phases.unsqueeze(0).expand(n_folds, -1).clone()
        dip = torch.zeros(n_points, dtype=torch.bool)
        dip[16:21] = True  # 5 consecutive points, phases 0.267-0.333
        assert dip.sum().item() == 5

        labels = torch.zeros(n_folds)
        labels[: n_folds // 2] = 1
        tokens = torch.randn(n_folds, n_points, 8) * 0.03
        tokens[: n_folds // 2, dip, 0] += 6.0
        mask = torch.ones(n_folds, n_points, dtype=torch.bool)

        encoder = FoldEncoder(
            d_model=8, n_heads=2, n_layers=1, dropout=0.0
        )
        classifier = nn.Linear(8, 1)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=0.03,
        )

        for _ in range(40):
            optimizer.zero_grad()
            logits = classifier(encoder(tokens, phases, mask)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()

        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            vectors, weights = encoder(
                tokens, phases, mask, return_weights=True
            )
            probabilities = classifier(vectors).squeeze(-1).sigmoid()

        assert probabilities[: n_folds // 2].mean() > 0.9
        assert probabilities[n_folds // 2 :].mean() < 0.1
        assert_close(weights.sum(dim=-1), torch.ones(n_folds))
        dip_weight = weights[: n_folds // 2, dip].sum(dim=-1)
        assert dip_weight.mean() > 0.5
        assert (dip_weight > 5 / 60).float().mean() >= 0.75
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
@pytest.mark.parametrize("grad_checkpoint", [False, True])
@pytest.mark.parametrize("encoder", ["rope", "cnn"])
def test_forward_backward(device, grad_checkpoint, encoder):
    batch = make_batch(device=device)
    model = FoldBranch(
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_cand_layers=2,
        fold_chunk=3,
        grad_checkpoint=grad_checkpoint,
        encoder=encoder,
        cnn_kwargs=CNN_KWARGS if encoder == "cnn" else None,
    ).to(device)
    result = model(batch)
    valid_scores = result["cand_scores"][batch["period_mask"]]
    loss = (
        result["z"].square().mean()
        + result["cand_vecs"][batch["period_mask"]].square().mean()
        + valid_scores.square().mean()
    )
    assert torch.isfinite(loss)
    loss.backward()

    for name, parameter in model.named_parameters():
        if "mask_embedding" in name:  # used only when the batch has hidden points
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
