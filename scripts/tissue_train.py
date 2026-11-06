"""evaluate and train the tissue classifier on the public Commons images

python scripts/fetch_commons.py            # downloads the images into data/commons
python scripts/tissue_train.py             # writes results/tecidos_*.csv and models/tissue_clf.npz

- images: the ones picked by eye in scripts/commons_curadoria.csv (H&E, normal tissue, real micrograph)
- test folds leave whole authors out: one author's series usually comes from the same slide
- answer = zero-shot (QuiltNet vs a description of each tissue) + logistic regression on the same embedding
- lab photos in --lab (one folder per tissue) are an external test, then join the final training
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
ap.add_argument("--lab", type=Path, default=Path("data/tecidos"))
ap.add_argument("--cache", type=Path, default=Path("data/commons_emb.npz"))
ap.add_argument("--out", type=Path, default=Path("models/tissue_clf.npz"))
ap.add_argument("--min-images", type=int, default=5)
a = ap.parse_args()

cred = pd.read_csv(a.commons / "creditos.csv").drop_duplicates(["tecido", "arquivo"]).rename(columns={"tecido": "pasta"})
df = pd.read_csv(a.curadoria).merge(cred, on=["pasta", "arquivo"], how="left")
df = df[[(a.commons / p / f).exists() for p, f in zip(df.pasta, df.arquivo)]]
df = df[df.tecido.map(df.tecido.value_counts()) >= a.min_images].reset_index(drop=True)
names = [n for n in NAMES if n in set(df.tecido)]
y = np.array([names.index(t) for t in df.tecido])
author = df.autor.fillna("").str.strip()
groups = author.where(author != "", df.titulo.fillna(df.arquivo)).str.lower().str[:40].values
print(f"{len(df)} images, {len(names)} tissues, {len(set(groups))} authors")

model = Model()
keys = np.array([f"{p}/{f}" for p, f in zip(df.pasta, df.arquivo)])
if a.cache.exists() and np.array_equal(np.load(a.cache)["keys"], keys):
    E = np.load(a.cache)["emb"]
else:
    E = model.emb([read_photo(a.commons / k) for k in keys])
    np.savez(a.cache, keys=keys, emb=E)


def fit(X, t):
    c = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X, t)
    coef = np.zeros((len(names), X.shape[1]), np.float32)
    inter = np.full(len(names), -30.0, np.float32)  # a tissue missing from the training fold gets ~0 probability
    coef[c.classes_], inter[c.classes_] = c.coef_, c.intercept_
    return {"coef": coef, "intercept": inter}


zs = model.zero_shot(E, names)
cl = np.zeros_like(zs)
for tr, te in GroupKFold(5).split(E, y, groups):
    cl[te] = clf_logp(fit(E[tr], y[tr]), E[te])
scores = {"zero-shot (sem treino)": zs, "classificador": cl, "zero-shot + classificador": zs + cl}


def metrics(name, s, t):
    top3 = np.argsort(-s, 1)[:, :3]
    present = sorted(set(t))
    return {"metodo": name, "n": len(t), "acertos_top1": int((s.argmax(1) == t).sum()),
            "top1": (s.argmax(1) == t).mean(), "top1_balanceada": np.mean([(s.argmax(1)[t == c] == c).mean() for c in present]),
            "acertos_top3": int((top3 == t[:, None]).any(1).sum()), "top3": (top3 == t[:, None]).any(1).mean(),
            "acaso_top1": 1 / len(names)}


rows = [metrics(k, s, y) for k, s in scores.items()]
best = zs + cl
per = pd.DataFrame({"tecido": [names[i] for i in y], "top1": best.argmax(1) == y,
                    "top3": (np.argsort(-best, 1)[:, :3] == y[:, None]).any(1)})
per = per.groupby("tecido").agg(n=("top1", "size"), acertos_top1=("top1", "sum"), top1=("top1", "mean"),
                                acertos_top3=("top3", "sum"), top3=("top3", "mean"))

lab = sorted(p for p in a.lab.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if a.lab.exists() else []
lab = [p for p in lab if p.parent.name in names]
if lab:
    LE = model.emb([read_photo(p) for p in lab])
    ly = np.array([names.index(p.parent.name) for p in lab])
    s = model.zero_shot(LE, names) + clf_logp(fit(E, y), LE)
    print("\nlab photos (other slides, through the eyepiece, never seen in training):")
    for p, t, g in zip(lab, ly, np.argsort(-s, 1)[:, :3]):
        print(f"  {p.name:20s} {names[t]:20s} -> {', '.join(names[i] for i in g)}{'' if g[0] == t else '   X'}")
    rows.append(metrics("zero-shot + classificador, fotos do laboratorio", s, ly))
    E, y = np.concatenate([E, LE]), np.concatenate([y, ly])

clf = fit(E, y)
a.out.parent.mkdir(exist_ok=True)
np.savez(a.out, names=np.array(names), **clf)
res = pd.DataFrame(rows)
print("\n" + res.to_string(index=False))
print("\n" + per.sort_values("top1").to_string())
res.to_csv("results/tecidos_eval.csv", index=False)
per.to_csv("results/tecidos_por_tecido.csv")
print(f"\nclassifier for {len(names)} tissues trained on {len(y)} images -> {a.out}")
