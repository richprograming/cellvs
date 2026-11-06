import numpy as np

from cellvs.evaluate import aggregate, l2norm


def build(emb, meta):
    """reference library: one l2-normed row per labeled treatment"""
    e, t = aggregate(emb, meta)
    keep = ((t.compound != "DMSO") & t.moa.notna()).values
    return l2norm(e[keep]), t[keep].reset_index(drop=True)


def threshold(lib_e, lib_t, q_emb, q_meta, pct=5):
    """similarity below which an answer is "unknown".
    calibrated on the library alone: how close known queries get to a treatment of another compound and week.
    pct=5 means ~5% of known queries get flagged by design"""
    sims = l2norm(q_emb) @ lib_e.T
    col = lambda df, c: df[c].to_numpy(dtype=object)  # pandas 3 string arrays do not broadcast to 2d
    ok = (col(q_meta, "compound")[:, None] != col(lib_t, "compound")[None]) & (col(q_meta, "batch")[:, None] != col(lib_t, "batch")[None])
    best = np.where(ok, sims, -np.inf).max(1)
    return float(np.percentile(best[np.isfinite(best)], pct))


def top_k(q_emb, lib_e, k=5):
    sims = l2norm(q_emb) @ lib_e.T
    idx = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(sims, idx, 1)
