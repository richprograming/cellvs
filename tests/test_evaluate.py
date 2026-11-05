import numpy as np
import pandas as pd

from cellvs.evaluate import aggregate, matches, nsc, nscb, per_moa


def fake_data(n_moa=4, comp_per_moa=3, batches=3, dim=16, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_moa, dim))
    rows, emb = [], []
    for m in range(n_moa):
        for c in range(comp_per_moa):
            for b in range(batches):
                rows.append(dict(compound=f"c{m}_{c}", conc=1.0, batch=f"w{b}", moa=f"moa{m}"))
                emb.append(centers[m] + noise * rng.normal(size=dim))
    return np.stack(emb), pd.DataFrame(rows)


def test_separable_clusters():
    emb, meta = fake_data()
    assert nsc(emb, meta)[0] > 0.95
    assert nscb(emb, meta)[0] > 0.95


def test_shuffled_labels_are_near_chance():
    emb, meta = fake_data(n_moa=4)
    meta["moa"] = np.random.default_rng(1).permutation(meta.moa.values)
    acc, _ = nsc(emb, meta)
    assert acc < 0.6  # chance is ~0.25


def test_nscb_skips_moa_seen_in_one_batch_only():
    emb, meta = fake_data(batches=1)
    acc, n = nscb(emb, meta)  # every treatment shares its batch with everything
    assert n == 0 and np.isnan(acc)


def test_neighbors_never_leak_compound_or_batch():
    emb, meta = fake_data()
    m = matches(emb, meta, exclude_batch=False)
    assert (meta.compound.values[m.i] != meta.compound.values[m.j]).all()
    m = matches(emb, meta, exclude_batch=True)
    assert (meta.compound.values[m.i] != meta.compound.values[m.j]).all()
    assert (meta.batch.values[m.i] != meta.batch.values[m.j]).all()


def test_per_moa_counts_match_overall():
    emb, meta = fake_data()
    t = per_moa(emb, meta)
    assert t.nsc_n.sum() == nsc(emb, meta)[1]
    assert t.nscb_n.sum() == nscb(emb, meta)[1]
    assert abs((t.nsc * t.nsc_n).sum() / t.nsc_n.sum() - nsc(emb, meta)[0]) < 1e-9


def test_aggregate_shape():
    emb, meta = fake_data()
    # duplicate every image so aggregate has something to average
    emb2 = np.concatenate([emb, emb])
    meta2 = pd.concat([meta, meta], ignore_index=True)
    a, am = aggregate(emb2, meta2)
    assert a.shape == emb.shape and len(am) == len(meta)
