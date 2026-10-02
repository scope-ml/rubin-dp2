# DP2 variable-star classifier: pretraining code

Self-supervised pretraining of a two-branch transformer on Rubin DP2 light curves, the first
stage of a periodic / aperiodic / non-variable classifier. Fine-tuning on Fritz labels is not
included here.

| Folder | Contents |
|---|---|
| `v1/` | the first full run (50 epochs, 327,335 objects); its own README describes it |
| `v2/` | the current version: same architecture, changed inputs, normalisation, ranking target and candidate periods |

## The model (both versions)

- **Folded branch:** each light curve is folded at every candidate period from the period search
  (LS, CE, AOV, FPW, MHF, ranks 1-50; about 130 per object in v1, about 190 in v2). Each fold is
  encoded by a transformer with rotary position embeddings on the phase angle (integer
  harmonics, so the phase zero point is irrelevant) and attention-pooled to one vector. A second
  transformer compares all candidates of an object and scores them; the top score is the
  model's period.
- **Unfolded branch:** a transformer on the time-ordered points with sinusoidal time embeddings.
- **Pretraining:** 30% of each object's observing nights are hidden; the model predicts the
  hidden magnitudes. Losses: unfolded prediction, a mixture over candidate folds weighted by the
  candidate scores, each fold's fraction of variance unexplained (FVU) on the hidden nights, and
  a ranking loss that teaches the scores to prefer folds that predict unseen nights well.

## What changed in v2, and why

1. **Scaling from the visible points only.** Each light curve is still divided by one scale
   shared by all bands, so a 0.005 mag star and a 1 mag star count equally in the losses
   (DP2 amplitudes span about 200 times). v2 uses the standard deviation, the usual choice for
   light-curve networks, but computes it **after** the nights are hidden, from the visible
   points only (`dataset.normalize_batch`). A scale computed from all points would tell the
   model how much variation the hidden nights contain, i.e. leak the answer.

2. **No asinh compression.** asinh kept the losses stable in v1 but flattened real shapes: it
   removed 35% of the range of RR Lyrae ab light curves and 62% of EA eclipse depths (measured on
   the labelled set), so peaks and eclipses counted less. asinh is meant for fluxes, not
   magnitudes; standard light-curve models do not compress. No clipping either: very deep points
   are usually real eclipses, and the spike filter already removes isolated bad points.

3. **Colours and amplitude as inputs** (`--side-dim 11`). Scaling removes the absolute
   amplitude, and the per-band median subtraction removes colour, yet both separate classes
   (large-amplitude RR Lyrae ab vs small RS CVn; blue delta Scuti vs red spotted stars). The
   model now gets 5 colours (u-g ... z-y), 5 missing-colour flags and log10 of the scale.
   Relative amplitudes between bands are kept in the data by the shared scale. In fine-tuning
   on v1 these inputs added about 0.03 macro-F1 each; in v2 they also enter pretraining, where
   the candidate scorer (the period choice) is trained.

4. **Absolute errors in the fold quality** (`--fvu-kind abs`). Without compression a few
   extreme points (deep eclipses falling only in hidden nights) would dominate squared errors
   and decide the period ranking. Absolute errors also match the Laplace likelihood used in the
   other losses. The loss test below confirms it: squared errors were the worst setting and
   their training loss jumped between batches.

5. **Conditional entropy in the ranking target** (`--ce-weight 1`). FVU rewards folds that
   predict the hidden nights well on average, which suits smooth shapes (RR Lyrae, EW,
   delta Scuti). Conditional entropy, the statistic behind the CE period search, rewards folds
   where points line up tightly in phase, which suits narrow features such as eclipses. It is
   computed the same cross-validated way as FVU: histogram from the visible points, scored on
   the hidden ones. Best setting in the loss test (+61 correct periods over absolute errors
   alone).

6. **Time embedding sized to DP2.** The unfolded branch's time features now span periods of
   0.005-300 d instead of 0.005-4000 d; DP2 baselines reach 238 d, so longer periods only
   wasted resolution. This is DP2-specific and would be widened for DR1.

7. **No photometric errors, as before.** On scanned non-variable stars the light-curve scatter
   is 4-7 times the quoted errors in g to z (u about 1.5), i.e. a missing ~0.01 mag floor.

8. **Clearer logging.** The ranking loss is also logged as a KL divergence (it can reach 0),
   together with the target's own entropy, which explains why the raw ranking loss sat near 5.2.

9. **Only exact duplicate candidate periods are removed.**
   v1 merged candidates whose frequencies differ by less than half a resolution element
   (0.5 / baseline) and kept the lowest frequency of each group. The kept period can then be off
   the true peak by enough to smear the fold: on the 824 scanned periodic objects, a candidate
   whose fold drifts by less than 0.1 cycle over the whole light curve existed for 767 objects
   before merging but only 578 after (189 lost, mostly RS CVn, RR Lyrae, EW and EA). Removing
   only identical values keeps 764 (about 194 candidates per object instead of 132, all under the
   256 limit). The cost is about 47% more folds, so training uses `--grad-checkpoint` (same
   results, less GPU memory). A period that several algorithms report identically is kept once.
   The old merging rule is removed from the v2 code; exact-duplicate removal is the only behaviour.

Not tested separately: the effect of the colour and amplitude inputs on pretraining itself
(only measured in fine-tuning), and each loss setting with more than one seed.

| | v1 | v2 |
|---|---|---|
| Scaling | robust spread (MAD) over all points, in the cache | standard deviation of the visible points, after masking |
| Compression | asinh | none |
| Side inputs | none | 5 colours, 5 missing flags, log amplitude |
| FVU | squared errors | absolute errors |
| Ranking target | softmax(-FVU / tau) | softmax(-(FVU + conditional entropy ratio) / tau) |
| Unfolded time embedding | 0.005-4000 d | 0.005-300 d |
| Candidate periods | merged within 0.5 / baseline (about 132 per object) | only exact duplicates removed (about 194) |

### Loss test behind the v2 choices

Four short runs (10,000 objects, 10 epochs, one GPU each), identical except the loss, all with the
v1 candidate merging (change 9 came after this test). Score: the
model's top period against the period on Fritz for 824 scanned periodic objects (within 2%;
half the period also counted for EW, EB and EA, whose Fritz period is the orbit).

| Run | Correct of 824 |
|---|---|
| absolute errors | 496 |
| squared errors | 311 |
| **absolute errors + conditional entropy (v2)** | **557** |
| squared errors + conditional entropy | 503 |
| v1 full run, for reference (327k objects, 50 epochs) | 560 |

The entropy term helps eclipsing binaries and RS CVn most, and costs a little on delta Scuti
(more picks at twice the period). Squared errors alone are unstable on this data.

## Running v2

```bash
# 1. cache: per-band median removed, no scaling, no asinh, side features, exact-duplicate candidates only
python build_cache.py --lc lc_parts/ --cands candidates.parquet --out cache_v2 --workers 64 \
    --mag-transform none --scale-kind none

# 2. pretrain (exact command in the docstring of train_pretrain.py)
torchrun --standalone --nnodes=1 --nproc_per_node=4 train_pretrain.py \
    --out runs/v2_full50 --cache cache_v2 --encoder rope --d-model 128 --n-heads 8 --epochs 50 \
    --mag-transform none --likelihood laplace --fvu-weight 1 --rank-weight 1 \
    --fvu-kind abs --side-dim 11 --normalize --ce-weight 1 --grad-checkpoint ...

# 3. period check against a table of known periods (oid, leaf, period)
python build_cache.py ... --ids truth_periods.csv --out cache_eval --mag-transform none --scale-kind none
python eval_periods.py cache_eval truth_periods.csv results.csv runs/v2_full50/best.pt

# 4. tests (142)
PYTHONPATH=. python -m pytest -q tests
```

The v2 options for scaling, compression, side inputs and losses default to the v1 behaviour; the
candidate periods are always de-duplicated by exact-duplicate removal.
Tested with Python 3.10, PyTorch 2.11, NumPy 1.23, pandas 2.3, pyarrow 24.

## v2 files

| File | Purpose |
|---|---|
| `dataset.py` | preprocessing, candidate-period de-duplication, side features, `normalize_batch`, batch padding |
| `build_cache.py` | preprocess all (or listed, `--ids`) objects once into .npz shards |
| `cached_dataset.py` | dataset reading the cache |
| `masking.py` | night-level masking |
| `fold_branch.py` | folded branch (phase-RoPE transformer, candidate comparison, side inputs) |
| `unfolded_branch.py` | unfolded branch |
| `cnn_encoder.py` | circular-padding CNN baseline for the folded branch |
| `pretrain.py` | pretraining model and losses, including the conditional entropy ratio |
| `train_pretrain.py` | single- and multi-GPU training, validation, checkpoints |
| `eval_periods.py` | top-period accuracy of checkpoints against known periods |
| `tests/` | unit tests (142) |
