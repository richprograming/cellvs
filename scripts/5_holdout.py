"""simulate new experiments: each week in turn is "new", the library is the other nine

new compound: the query's own compound is also out of the library, so it has to be recognized by similarity
class removed: the query's whole moa is out of the library, the right answer is "unknown"
no dmso correction, as in predict.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellvs.evaluate import l2norm
from cellvs.library import build, threshold

meta = pd.read_csv("data/metadata.csv")
labeled = meta.moa.notna().values
rows = []

for b in ["resnet50", "dinov2_vits14"]:
    raw = np.load(f"results/emb_{b}.npy")
    lib_e, lib_t = build(raw, meta)
    levels = {"tratamento": (lib_e, lib_t), "imagem": (l2norm(raw[labeled]), meta[labeled].reset_index(drop=True))}
    for level, (qe, qm) in levels.items():
        for w in sorted(meta.batch.unique()):
            in_lib = (lib_t.batch != w).values
            cal = (qm.batch != w).values
            thr = threshold(lib_e[in_lib], lib_t[in_lib], qe[cal], qm[cal])
            for i in np.where((qm.batch == w).values)[0]:
                s = lib_e @ qe[i]
                new_comp = in_lib & (lib_t.compound != qm.compound[i]).values
                no_class = new_comp & (lib_t.moa != qm.moa[i]).values
                order = np.where(new_comp)[0][np.argsort(-s[new_comp], kind="stable")]
                rows.append({
                    "backbone": b, "level": level, "week": w, "moa": qm.moa[i],
                    "known": bool((lib_t.moa.values[new_comp] == qm.moa[i]).any()),
                    "hit": lib_t.moa[order[0]] == qm.moa[i],
                    "hit_top5": bool((lib_t.moa.values[order[:5]] == qm.moa[i]).any()),
                    "sim": float(s[order[0]]),
                    "sim_no_class": float(s[no_class].max()),
                    "thr": thr,
                    "chance": float((lib_t.moa.values[new_comp] == qm.moa[i]).mean()),
                })

r = pd.DataFrame(rows)
r.to_csv("results/holdout_queries.csv", index=False)

summary = []
for (b, level), g in r.groupby(["backbone", "level"], sort=False):
    k = g[g.known]
    flagged = k.sim < k.thr
    summary.append({
        "backbone": b, "level": level, "queries_known": len(k),
        "top1": k.hit.mean(), "top1_hits": int(k.hit.sum()),
        "top1_balanced": k.groupby("moa").hit.mean().mean(),
        "top5": k.hit_top5.mean(), "chance": k.chance.mean(),
        "flagged_unknown_known": flagged.mean(),
        "top1_when_answered": k.hit[~flagged].mean(),
        "flagged_unknown_class_removed": (g.sim_no_class < g.thr).mean(),
        "queries_class_removed": len(g),
        "flagged_unknown_absent_class": (g[~g.known].sim < g[~g.known].thr).mean() if (~g.known).any() else np.nan,
        "queries_absent_class": int((~g.known).sum()),
        "auroc_known_vs_removed": roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(g))], np.r_[g.sim, g.sim_no_class]),
    })
summary = pd.DataFrame(summary)
summary.to_csv("results/holdout_summary.csv", index=False)
per_moa = r[r.known].groupby(["backbone", "level", "moa"]).agg(top1=("hit", "mean"), n=("hit", "size")).reset_index()
per_moa.to_csv("results/holdout_per_moa.csv", index=False)
print(summary.T.to_string())
