import numpy as np
import pandas as pd


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def aggregate(emb, meta):
    """mean embedding per (compound, conc, batch) -> one row per treatment"""
    keys = ["compound", "conc", "batch"]
    df = meta[keys + ["moa"]].copy()
    df["i"] = np.arange(len(df))
    rows, out = [], []
    for k, g in df.groupby(keys, dropna=False):
        out.append(emb[g.i.values].mean(0))
        rows.append({**dict(zip(keys, k)), "moa": g.moa.iloc[0]})
    return np.stack(out), pd.DataFrame(rows)


def center_on_dmso(emb, meta):
    """subtract the DMSO mean of each batch. crude, TVN would be better (see notes.md)"""
    emb = emb.copy()
    for b in meta.batch.unique():
        in_b = (meta.batch == b).values
        ctrl = in_b & (meta.compound == "DMSO").values
        if ctrl.any():
            emb[in_b] -= emb[ctrl].mean(0)
    return emb


def matches(emb, meta, exclude_batch):
    """one row per evaluated treatment: its nearest neighbor, hit, and hit chance if neighbors were random"""
    emb = l2norm(emb)
    sim = emb @ emb.T
    comp = meta.compound.values
    moa = meta.moa.values
    batch = meta.batch.values

    rows = []
    for i in range(len(meta)):
        bad = comp == comp[i]
        if exclude_batch:
            bad = bad | (batch == batch[i])
        ok = ~bad
        # moa only exists in this compound/batch -> nothing to match against, skip
        if not (moa[ok] == moa[i]).any():
            continue
        j = np.where(ok)[0][np.argmax(sim[i, ok])]
        rows.append({
            "i": i, "j": j, "moa": moa[i],
            "hit": bool(moa[j] == moa[i]),
            "chance": float((moa[ok] == moa[i]).mean()),
        })
    return pd.DataFrame(rows, columns=["i", "j", "moa", "hit", "chance"])


def _acc(emb, meta, exclude_batch):
    m = matches(emb, meta, exclude_batch)
    return float(m.hit.mean()) if len(m) else float("nan"), len(m)


def nsc(emb, meta):
    """not-same-compound 1-NN accuracy, cosine"""
    return _acc(emb, meta, exclude_batch=False)


def nscb(emb, meta):
    """not-same-compound-and-batch 1-NN accuracy, cosine"""
    return _acc(emb, meta, exclude_batch=True)


def per_moa(emb, meta):
    out = []
    for name, excl in [("nsc", False), ("nscb", True)]:
        g = matches(emb, meta, excl).groupby("moa")
        out.append(pd.DataFrame({
            f"{name}": g.hit.mean(),
            f"{name}_n": g.hit.size(),
            f"{name}_chance": g.chance.mean(),
        }))
    return pd.concat(out, axis=1)
