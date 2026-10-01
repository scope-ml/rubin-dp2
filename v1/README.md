# DP2 variable-star classifier: pretraining code

Self-supervised pretraining of a two-branch transformer on Rubin DP2 light curves, the first
stage of a periodic / aperiodic / non-variable classifier. Fine-tuning on Fritz labels comes
next and is not included here.

## Pipeline

1. **Preprocessing** (`dataset.py`, `build_cache.py`, `cached_dataset.py`)
   - per-band spike filter (1 mag against the local 3-point median) and 5-minute cadence filter;
   - per-band median subtraction, one robust scale shared by all bands, then asinh;
   - candidate periods: all ranks (up to 50) of LS, CE, AOV, FPW and MHF from the period
     search, de-duplicated to the baseline's frequency resolution (typically 100-200 per object);
   - objects with fewer than 20 points or no valid candidate are skipped.
   `build_cache.py` writes everything once to .npz shards; training then reads the cache.

2. **Model** (`fold_branch.py`, `unfolded_branch.py`, `cnn_encoder.py`)
   - **Folded branch:** the light curve is folded at every candidate period. Each fold is
     encoded by a transformer with rotary position embeddings on the phase angle (integer
     harmonics, so attention depends only on phase differences and the phase zero point is
     irrelevant), then attention-pooled to one vector. A second transformer compares all
     candidates of an object (each with log period and log cycles covered) and scores them.
   - **Unfolded branch:** a transformer on the time-ordered points with sinusoidal time
     embeddings, attention-pooled to one vector.
   - **Baseline:** `encoder="cnn"` replaces the fold transformer with a 1D ResNet over phase bins
     with circular padding; everything else stays the same.

3. **Pretraining** (`masking.py`, `pretrain.py`, `train_pretrain.py`)
   30% of each object's observing nights are hidden; the model predicts the hidden magnitudes.
   Losses: unfolded prediction (L1), a mixture over candidate folds weighted by the candidate
   scores (Laplace likelihood), the fraction of variance unexplained of every fold on the hidden
   nights, and a ranking loss that teaches the scores to prefer folds that predict unseen nights
   well. See the docstring of `pretrain.py`.

## Running

```bash
# 1. build the cache
python build_cache.py --lc lc_parts/ --cands candidates.parquet --out cache_full --workers 16

# 2. pretrain (the exact settings of the 50-epoch run are in the docstring of train_pretrain.py)
torchrun --standalone --nnodes=1 --nproc_per_node=4 train_pretrain.py \
    --out runs/full50 --cache cache_full --encoder rope --d-model 128 --n-heads 8 --epochs 50 ...

# 3. tests
PYTHONPATH=. python -m pytest -q tests
```

Tested with Python 3.10, PyTorch 2.11, NumPy 1.23, pandas 2.3, pyarrow 24.

## Full run

327,335 DP2 light curves (956 skipped: too few points or no candidates), 50 epochs, validation
on a fixed 10% of objects. Best validation loss 7.998 at epoch 47.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | preprocessing, in-memory dataset, batch padding |
| `build_cache.py` | preprocess all objects once into .npz shards |
| `cached_dataset.py` | dataset reading the cache |
| `masking.py` | night-level masking |
| `fold_branch.py` | folded branch (phase-RoPE transformer, candidate comparison) |
| `unfolded_branch.py` | unfolded branch |
| `cnn_encoder.py` | circular-padding CNN baseline for the folded branch |
| `pretrain.py` | pretraining model and losses |
| `train_pretrain.py` | single- and multi-GPU training, validation, checkpoints |
| `tests/` | unit tests (82) |
