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


def _one_nn_acc(emb, meta, exclude_batch):
    emb = l2norm(emb)
    sim = emb @ emb.T
    comp = meta.compound.values
    moa = meta.moa.values
    batch = meta.batch.values

    hits, n = 0, 0
    for i in range(len(meta)):
        bad = comp == comp[i]
        if exclude_batch:
            bad = bad | (batch == batch[i])
        ok = ~bad
        # moa only exists in this compound/batch -> nothing to match against, skip
        if not (moa[ok] == moa[i]).any():
            continue
        j = np.where(ok)[0][np.argmax(sim[i, ok])]
        hits += moa[j] == moa[i]
        n += 1
    return hits / n if n else float("nan"), n


def nsc(emb, meta):
    """not-same-compound 1-NN accuracy, cosine"""
    return _one_nn_acc(emb, meta, exclude_batch=False)


def nscb(emb, meta):
    """not-same-compound-and-batch 1-NN accuracy, cosine"""
    return _one_nn_acc(emb, meta, exclude_batch=True)
