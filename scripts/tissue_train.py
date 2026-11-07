"""evaluate and train the tissue classifier on public images: Wikimedia Commons + GTEx slides

python scripts/fetch_commons.py            # downloads the Commons images into data/commons
python scripts/fetch_gtex.py               # crops of GTEx slides into data/gtex (optional)
python scripts/tissue_train.py             # writes results/tecidos_*.csv and models/tissue_clf.npz

- Commons images: the ones picked by eye in scripts/commons_curadoria.csv (H&E, normal tissue, real micrograph)
- test folds leave whole authors (Commons) or whole donors (GTEx) out: one author's series or one donor's
  slide shares everything but the tissue, and leaving it on both sides would inflate the result
- answer = zero-shot (QuiltNet vs a description of each tissue) + logistic regression on the same embedding
- in training, each GTEx slide weighs as much as one Commons image, however many crops it gave
- the comparisons below were fixed before looking at any GTEx result:
    A. Commons, classifier trained on Commons only (the previous model)
    B. Commons, classifier trained on Commons + GTEx
    C. GTEx, trained on Commons only vs on Commons + GTEx
    D. lab photos (other slides, phone through the eyepiece), same two models
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellvs.histo import NAMES, Model, clf_logp, read_photo

ap = argparse.ArgumentParser()
ap.add_argument("--commons", type=Path, default=Path("data/commons"))
ap.add_argument("--curadoria", type=Path, default=Path("scripts/commons_curadoria.csv"))
ap.add_argument("--gtex", type=Path, default=Path("data/gtex"))
ap.add_argument("--lab", type=Path, default=Path("data/tecidos"))
ap.add_argument("--cache", type=Path, default=Path("data/emb_cache.npz"))
ap.add_argument("--out", type=Path, default=Path("models/tissue_clf.npz"))
ap.add_argument("--min-images", type=int, default=5)
a = ap.parse_args()

# --- data ---------------------------------------------------------------------------------------
cred = pd.read_csv(a.commons / "creditos.csv").drop_duplicates(["tecido", "arquivo"]).rename(columns={"tecido": "pasta"})
cm = pd.read_csv(a.curadoria).merge(cred, on=["pasta", "arquivo"], how="left")
cm = cm[[(a.commons / p / f).exists() for p, f in zip(cm.pasta, cm.arquivo)]].reset_index(drop=True)
author = cm.autor.fillna("").str.strip()
cm["grupo"] = "commons:" + author.where(author != "", cm.titulo.fillna(cm.arquivo)).str.lower().str[:40]
cm["path"] = [str(a.commons / p / f) for p, f in zip(cm.pasta, cm.arquivo)]
cm["peso"] = 1.0

gt = pd.DataFrame(columns=["tecido", "path", "grupo", "peso"])
if (a.gtex / "creditos.csv").exists():
    gt = pd.read_csv(a.gtex / "creditos.csv")
    gt = gt[[(a.gtex / t / f).exists() for t, f in zip(gt.tecido, gt.arquivo)]].reset_index(drop=True)
    gt["path"] = [str(a.gtex / t / f) for t, f in zip(gt.tecido, gt.arquivo)]
    gt["grupo"] = "gtex:" + gt.doador
    gt["peso"] = 1.0 / gt.groupby("slide").slide.transform("size")

# a tissue enters when Commons has enough images of it or GTEx has it
counts = cm.tecido.value_counts()
names = [n for n in NAMES if counts.get(n, 0) >= a.min_images or n in set(gt.tecido)]
cm = cm[cm.tecido.isin(names)].reset_index(drop=True)
print(f"Commons: {len(cm)} images, {cm.grupo.nunique()} authors | GTEx: {len(gt)} crops, "
      f"{gt.slide.nunique() if len(gt) else 0} slides, {gt.grupo.nunique()} donors | {len(names)} tissues")

model = Model()


def embed_all(paths):
    """embeddings with a cache keyed by path, so new images only cost their own forward pass"""
    cache = dict(np.load(a.cache, allow_pickle=True)["d"].item()) if a.cache.exists() else {}
    todo = [p for p in paths if p not in cache]
    for i in range(0, len(todo), 64):
        chunk = todo[i : i + 64]
        for p, e in zip(chunk, model.emb([read_photo(p) for p in chunk])):
            cache[p] = e
        print(f"  embeddings {min(i + 64, len(todo))}/{len(todo)}", flush=True)
        if (i // 64) % 10 == 9:
            np.savez(a.cache, d=np.array(cache, dtype=object))
    if todo:
        np.savez(a.cache, d=np.array(cache, dtype=object))
    return np.stack([cache[p] for p in paths]) if paths else np.zeros((0, model.text.shape[1]), np.float32)


Ec, yc = embed_all(list(cm.path)), np.array([names.index(t) for t in cm.tecido])
Eg, yg = embed_all(list(gt.path)), np.array([names.index(t) for t in gt.tecido], dtype=int)


def fit(X, t, w):
    c = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X, t, sample_weight=w)
    coef = np.zeros((len(names), X.shape[1]), np.float32)
    inter = np.full(len(names), -30.0, np.float32)  # a tissue missing from the training fold gets ~0 probability
    coef[c.classes_], inter[c.classes_] = c.coef_, c.intercept_
    return {"coef": coef, "intercept": inter}


def metrics(test, method, s, t):
    top3 = np.argsort(-s, 1)[:, :3]
    present = sorted(set(t))
    return {"teste": test, "metodo": method, "n": len(t), "acertos_top1": int((s.argmax(1) == t).sum()),
            "top1": (s.argmax(1) == t).mean(), "top1_balanceada": np.mean([(s.argmax(1)[t == c] == c).mean() for c in present]),
            "acertos_top3": int((top3 == t[:, None]).any(1).sum()), "top3": (top3 == t[:, None]).any(1).mean(),
            "acaso_top1": 1 / len(present)}


rows = []
wc, wg = cm.peso.values, gt.peso.values.astype(float)

# --- A, B: Commons test folds, by author --------------------------------------------------------
zs_c = model.zero_shot(Ec, names)
clA, clB = np.zeros_like(zs_c), np.zeros_like(zs_c)
for tr, te in GroupKFold(5).split(Ec, yc, cm.grupo):
    clA[te] = clf_logp(fit(Ec[tr], yc[tr], wc[tr]), Ec[te])
    if len(gt):
        clB[te] = clf_logp(fit(np.concatenate([Ec[tr], Eg]), np.concatenate([yc[tr], yg]), np.concatenate([wc[tr], wg])), Ec[te])
rows += [metrics("Commons", "zero-shot (sem treino)", zs_c, yc),
         metrics("Commons", "A. zero-shot + classificador (Commons)", zs_c + clA, yc)]
if len(gt):
    rows.append(metrics("Commons", "B. zero-shot + classificador (Commons + GTEx)", zs_c + clB, yc))

# --- C: GTEx test folds, by donor ---------------------------------------------------------------
if len(gt):
    zs_g = model.zero_shot(Eg, names)
    cl_old = clf_logp(fit(Ec, yc, wc), Eg)
    cl_new = np.zeros_like(zs_g)
    for tr, te in GroupKFold(5).split(Eg, yg, gt.grupo):
        cl_new[te] = clf_logp(fit(np.concatenate([Ec, Eg[tr]]), np.concatenate([yc, yg[tr]]), np.concatenate([wc, wg[tr]])), Eg[te])
    rows += [metrics("GTEx", "zero-shot (sem treino)", zs_g, yg),
             metrics("GTEx", "A. zero-shot + classificador (Commons)", zs_g + cl_old, yg),
             metrics("GTEx", "B. zero-shot + classificador (Commons + GTEx)", zs_g + cl_new, yg)]

# --- D: lab photos ------------------------------------------------------------------------------
lab = sorted(p for p in a.lab.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if a.lab.exists() else []
lab = [p for p in lab if p.parent.name in names]
final_X, final_y, final_w = np.concatenate([Ec, Eg]), np.concatenate([yc, yg]), np.concatenate([wc, wg])
if lab:
    LE = model.emb([read_photo(p) for p in lab])
    ly = np.array([names.index(p.parent.name) for p in lab])
    zs_l = model.zero_shot(LE, names)
    rows.append(metrics("fotos do laboratorio", "A. zero-shot + classificador (Commons)", zs_l + clf_logp(fit(Ec, yc, wc), LE), ly))
    s = zs_l + clf_logp(fit(final_X, final_y, final_w), LE)
    if len(gt):
        rows.append(metrics("fotos do laboratorio", "B. zero-shot + classificador (Commons + GTEx)", s, ly))
    print("\nlab photos (other slides, through the eyepiece, never seen in training):")
    for p, t, g in zip(lab, ly, np.argsort(-s, 1)[:, :3]):
        print(f"  {p.name:20s} {names[t]:20s} -> {', '.join(names[i] for i in g)}{'' if g[0] == t else '   X'}")
    p = np.exp(s - s.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    # stays in data/: these are the lab's own photos
    pd.DataFrame([{"caminho": str(f), "tecido": names[t], **{f"top{k + 1}": names[i] for k, i in enumerate(np.argsort(-q)[:3])},
                   **{f"p{k + 1}": float(q[i]) for k, i in enumerate(np.argsort(-q)[:3])}}
                  for f, t, q in zip(lab, ly, p)]).to_csv("data/tecidos_teste_externo.csv", index=False)
    final_X, final_y, final_w = np.concatenate([final_X, LE]), np.concatenate([final_y, ly]), np.concatenate([final_w, np.ones(len(ly))])

# --- per tissue: best available model on each test set -----------------------------------------
best = [("Commons", cm.tecido, zs_c + (clB if len(gt) else clA), yc)]
if len(gt):
    best.append(("GTEx", gt.tecido, zs_g + cl_new, yg))
per = []
for test, tec, s, t in best:
    d = pd.DataFrame({"tecido": tec.values, "top1": s.argmax(1) == t, "top3": (np.argsort(-s, 1)[:, :3] == t[:, None]).any(1)})
    d = d.groupby("tecido").agg(n=("top1", "size"), acertos_top1=("top1", "sum"), top1=("top1", "mean"),
                                acertos_top3=("top3", "sum"), top3=("top3", "mean")).reset_index()
    per.append(d.assign(teste=test))
per = pd.concat(per)[["teste", "tecido", "n", "acertos_top1", "top1", "acertos_top3", "top3"]]

clf = fit(final_X, final_y, final_w)
a.out.parent.mkdir(exist_ok=True)
np.savez(a.out, names=np.array(names), **clf)
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n" + res.to_string(index=False))
print("\n" + per.sort_values(["teste", "top1"]).to_string(index=False))
res.to_csv("results/tecidos_eval.csv", index=False)
per.to_csv("results/tecidos_por_tecido.csv", index=False)
print(f"\nclassifier for {len(names)} tissues trained on {len(final_y)} images -> {a.out}")
