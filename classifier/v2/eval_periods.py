"""Period check: for each object with a known period (a CSV with columns oid, leaf, period),
is a pretrained model's top-scored candidate that period?
Within 2%; also 2P and P/2; and whether the scanned period is among the candidates at all.
Usage: python eval_periods.py EVAL_CACHE TRUTH_CSV OUT_CSV CKPT [CKPT ...]"""
import sys
import numpy as np, pandas as pd, torch
from cached_dataset import CachedLightCurveDataset
from dataset import collate, normalize_batch
from pretrain import PretrainModel

cache, truth, out = sys.argv[1:4]
T = pd.read_csv(truth).dropna(subset=["period"]).set_index("oid")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
near = lambda x, y, f=1.0: np.abs(x / (y * f) - 1) <= 0.02
summary = []
for ckpt in sys.argv[4:]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    model = PretrainModel(encoder=a["encoder"], d_model=a["d_model"], likelihood=a["likelihood"],
        fvu_weight=a["fvu_weight"], rank_weight=a["rank_weight"], rank_tau=a["rank_tau"], fvu_cap=a["fvu_cap"],
        fvu_kind=a.get("fvu_kind", "sq"), side_dim=a.get("side_dim", 0), normalize=a.get("normalize", False),
        fold_kwargs=dict(n_heads=a["n_heads"], n_layers=a["fold_layers"], n_cand_layers=a["cand_layers"],
                         n_harm=a["n_harm"], fold_chunk=a["fold_chunk"], grad_checkpoint=False),
        unfolded_kwargs=dict(n_heads=a["n_heads"], n_layers=a["unf_layers"], grad_checkpoint=False))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected and all("ce" in k for k in missing), (missing, unexpected)
    model.to(dev).eval()
    ds = CachedLightCurveDataset(cache, max_points=a.get("max_points", 512), max_periods=a.get("max_periods", 256), train=False)
    rows = []
    for j in range(0, len(ds), 16):
        b = {k: v.to(dev) for k, v in collate([ds[i] for i in range(j, min(j + 16, len(ds)))]).items()}
        if a.get("normalize"):
            b = normalize_batch(b, None)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            s = model.fold(b)["cand_scores"].float()
        s = s.masked_fill(~b["period_mask"], -1e9)
        top = s.argmax(1)
        for r in range(len(b["id"])):
            oid = int(b["id"][r])
            if oid not in T.index:
                continue
            P = b["periods"][r][b["period_mask"][r]].cpu().numpy()
            rows.append(dict(oid=oid, leaf=T.loc[oid, "leaf"], P_true=T.loc[oid, "period"],
                             top=float(b["periods"][r, top[r]]), in_cands=bool(near(P, T.loc[oid, "period"]).any())))
    R = pd.DataFrame(rows)
    R["exact"] = near(R.top, R.P_true); R["twoP"] = near(R.top, R.P_true, 2); R["halfP"] = near(R.top, R.P_true, .5)
    R["run"] = ckpt.split("/")[-2]; R["epoch"] = ck.get("epoch")
    summary.append(R)
    g = R.groupby("leaf").agg(n=("oid", "size"), exact=("exact", "sum"), twoP=("twoP", "sum"), halfP=("halfP", "sum"), in_cands=("in_cands", "sum"))
    print(f"\n== {R.run[0]} (epoch {ck.get('epoch')}, fvu {a.get('fvu_kind')}, likelihood {a['likelihood']}, ce_weight {a.get('ce_weight', 0)})")
    print(g.to_string()); print(f"TOTAL exact {int(R.exact.sum())}/{len(R)}  2P {int(R.twoP.sum())}  P/2 {int(R.halfP.sum())}  ceiling {int(R.in_cands.sum())}")
pd.concat(summary).to_csv(out, index=False)
