"""is the score real, or does the week give it away?

week_only: no images, neighbor drawn at random from the same week
dmso_fingerprint: each treatment replaced by the DMSO mean of its week, no treatment signal at all
shuffle_within_week: moa shuffled between compounds of a week, keeps which classes each week has
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellvs.evaluate import aggregate, center_on_dmso, l2norm, matches

N_PERM, N_BOOT = 200, 2000
meta = pd.read_csv("data/metadata.csv")
is_dmso = (meta.compound == "DMSO").values


def treatments(emb):
    e, t = aggregate(emb, meta)
    keep = ((t.compound != "DMSO") & t.moa.notna()).values
    return e[keep], t[keep].reset_index(drop=True)


def week_only(t):
    comp, moa, batch = t.compound.values, t.moa.values, t.batch.values
    out = []
    for i in range(len(t)):
        pool = (comp != comp[i]) & (batch == batch[i])
        if pool.any():
            out.append((moa[pool] == moa[i]).mean())
    return float(np.mean(out)), len(out), len(t)


def dmso_fingerprint(raw, t):
    centers = {b: l2norm(raw[(meta.batch == b).values & is_dmso].mean(0, keepdims=True))[0] for b in meta.batch.unique()}
    comp, moa, batch = t.compound.values, t.moa.values, t.batch.values
    out = []
    for i in range(len(t)):
        pool = (comp != comp[i]) & (batch != batch[i])
        if not (moa[pool] == moa[i]).any():
            continue
        weeks = sorted(set(batch[pool]), key=lambda w: -(centers[w] @ centers[batch[i]]))
        nearest = pool & (batch == weeks[0])
        out.append((moa[nearest] == moa[i]).mean())
    return float(np.mean(out))


def shuffle_within_week(e, t, exclude_batch, rng):
    accs = []
    for _ in range(N_PERM):
        s = t.copy()
        for b in t.batch.unique():
            # taxol sits in all 10 weeks, so it keeps its label
            comps = t[(t.batch == b) & (t.compound != "taxol")].groupby("compound").moa.first()
            new = dict(zip(comps.index, rng.permutation(comps.values)))
            on = (t.batch == b) & (t.compound != "taxol")
            s.loc[on, "moa"] = t.compound[on].map(new)
        accs.append(matches(e, s, exclude_batch).hit.mean())
    return float(np.mean(accs))


def boot_ci(hits_by_compound, rng):
    comps = list(hits_by_compound)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(comps), len(comps))
        vals.append(np.concatenate([hits_by_compound[comps[k]] for k in pick]).mean())
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def hits(e, t, exclude_batch):
    m = matches(e, t, exclude_batch)
    h = m.assign(compound=t.compound.values[m.i]).groupby("compound").hit
    return {c: g.values.astype(float) for c, g in h}


def paired_ci(a, b, rng):
    comps = sorted(set(a) & set(b))
    d = {c: a[c] - b[c] for c in comps}
    return boot_ci(d, rng)


def sphere(emb):
    # exploratory: whitening fitted on DMSO only, 99% of variance, never sees moa
    plate_centered = center_on_dmso(emb, meta.assign(batch=meta.plate))
    return PCA(0.99, whiten=True, random_state=0).fit(plate_centered[is_dmso]).transform(plate_centered)


rng = np.random.default_rng(0)
rows, hit_store, summary = [], {}, {}
_, t0 = treatments(np.load("results/emb_resnet50.npy"))
summary["week_only_nsc"], summary["week_only_n"], summary["n_treatments"] = week_only(t0)

for b in ["resnet50", "dinov2_vits14"]:
    raw = np.load(f"results/emb_{b}.npy")
    summary[f"dmso_fingerprint_nscb_{b}"] = dmso_fingerprint(raw, t0)
    variants = {
        "raw": raw,
        "centered_week": center_on_dmso(raw, meta),
        "centered_plate": center_on_dmso(raw, meta.assign(batch=meta.plate)),
        "sphered_plate": sphere(raw),
    }
    for name, emb in variants.items():
        e, t = treatments(emb)
        no_ctrl = ~((t.compound == "taxol") & (t.groupby(["compound", "conc"]).batch.transform("nunique") > 1)).values
        row = {"backbone": b, "variant": name}
        for metric, excl in [("nsc", False), ("nscb", True)]:
            m = matches(e, t, excl)
            h = hits(e, t, excl)
            hit_store[b, name, metric] = h
            low, high = boot_ci(h, rng)
            row |= {
                metric: float(m.hit.mean()), f"{metric}_hits": int(m.hit.sum()), f"{metric}_n": len(m),
                f"{metric}_ci_low": low, f"{metric}_ci_high": high,
                f"{metric}_shuffle_within_week": shuffle_within_week(e, t, excl, rng),
                f"{metric}_no_taxol_control": float(matches(e[no_ctrl], t[no_ctrl].reset_index(drop=True), excl).hit.mean()),
            }
        rows.append(row)

for metric in ["nsc", "nscb"]:
    for label, x, y in [
        ("dinov2-resnet50 raw", ("dinov2_vits14", "raw"), ("resnet50", "raw")),
        ("dinov2-resnet50 centered_week", ("dinov2_vits14", "centered_week"), ("resnet50", "centered_week")),
        ("raw-centered_week resnet50", ("resnet50", "raw"), ("resnet50", "centered_week")),
        ("raw-centered_week dinov2", ("dinov2_vits14", "raw"), ("dinov2_vits14", "centered_week")),
    ]:
        summary[f"diff_ci_{metric} {label}"] = paired_ci(hit_store[(*x, metric)], hit_store[(*y, metric)], rng)

pd.DataFrame(rows).to_csv("results/leak_checks.csv", index=False)
Path("results/leak_checks_summary.json").write_text(json.dumps(summary, indent=2))
print(pd.DataFrame(rows).T.to_string())
print(json.dumps(summary, indent=2))
