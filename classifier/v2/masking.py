# masking.py
"""Night-level masking for self-supervised pretraining.

Points are grouped into observing nights (a new night starts after a gap longer than
`night_gap` days). A fixed fraction of nights (30% by default) is hidden per object, and the
model must predict the hidden magnitudes. Hiding whole nights, not single points, prevents the
model from copying a neighbouring point taken minutes earlier.
"""
import torch


def night_mask(
    t: torch.Tensor,
    point_mask: torch.Tensor,
    generator: torch.Generator,
    frac: float = 0.30,
    night_gap: float = 0.5,
    min_visible_nights: int = 3,
    min_visible_points: int = 10,
) -> torch.Tensor:
    """Select whole observing nights to hide, independently for each object.

    Returns a [B, N] boolean tensor, True for hidden points. At least `min_visible_nights`
    nights and `min_visible_points` points always stay visible; objects too small to satisfy
    both are left fully visible.
    """
    if t.shape != point_mask.shape or t.ndim != 2:
        raise ValueError("t and point_mask must have matching [B, N] shapes")
    if not 0 <= frac <= 1:
        raise ValueError("frac must be between 0 and 1")
    if night_gap < 0 or min_visible_nights < 0 or min_visible_points < 0:
        raise ValueError("night_gap and visibility floors must be nonnegative")

    hidden = torch.zeros_like(point_mask, dtype=torch.bool)
    for row in range(t.shape[0]):
        real = point_mask[row].nonzero(as_tuple=True)[0]
        if real.numel() == 0:
            continue

        # Label each point with its night: a new night starts after a gap > night_gap days.
        times = t[row, real]
        starts = torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=t.device),
                times[1:] - times[:-1] > night_gap,
            )
        )
        night_ids = starts.cumsum(dim=0) - 1
        n_nights = int(night_ids[-1].item()) + 1
        target = min(
            round(frac * n_nights),
            max(0, n_nights - min_visible_nights),
        )
        if target == 0 or real.numel() < min_visible_points:
            continue

        # Visit nights in random order and hide them while enough points remain visible.
        order = torch.randperm(n_nights, generator=generator).tolist()
        visible_points = int(real.numel())
        selected = []
        for night in order:
            count = int((night_ids == night).sum().item())
            if visible_points - count >= min_visible_points:
                selected.append(night)
                visible_points -= count
                if len(selected) == target:
                    break

        for night in selected:
            hidden[row, real[night_ids == night]] = True

    return hidden
