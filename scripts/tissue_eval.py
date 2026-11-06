"""how often the tissue is right on images the method never saw

python scripts/tissue_eval.py --commons data/commons --lab data/tecidos
- library = public Commons images (scripts/fetch_commons.py); test folds leave whole authors out,
  because one author's series usually comes from the same slide
- external test = the lab photos, one folder per tissue, scored by a classifier fitted on all Commons images
writes results/tecidos_eval.csv, results/tecidos_por_tecido.csv and models/tissue_clf.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellvs.histo import TISSUES, Model, read_photo

NAMES = list(TISSUES)
KIND = ["an H&E stained histology section", "a drawing or a diagram", "a figure with text labels and arrows",
        "an immunohistochemistry stain in brown", "a fluorescence microscopy image", "a silver stain in black",
        "a photograph of an animal or a whole organ"]

ap = argparse.ArgumentParser()
ap.add_argument("--commons", type=Path, default=Path("data/commons"))
ap.add_argument("--lab", type=Path, default=Path("data/tecidos"))
ap.add_argument("--cache", type=Path, default=Path("data/commons_feats.npz"))
ap.add_argument("--curadoria", type=Path, default=Path("scripts/commons_curadoria.csv"), help="images picked by eye")
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()

cred = pd.read_csv(a.commons / "creditos.csv").drop_duplicates(["tecido", "arquivo"])
cred = cred[[(a.commons / t / f).exists() for t, f in zip(cred.tecido, cred.arquivo)]].reset_index(drop=True)
lab = sorted(p for p in a.lab.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if a.lab.exists() else []
model = Model()


def feats(paths):
    q, d = [], []
    for s in range(0, len(paths), 16):
        rgbs = [read_photo(p) for p in paths[s : s + 16]]
        q.append(model.quilt_emb(rgbs))
        d.append(model.similarity_emb(rgbs))
        print(f"  {min(s + 16, len(paths))}/{len(paths)}", end="\r", flush=True)
    return np.concatenate(q), np.concatenate(d)


paths = [a.commons / t / f for t, f in zip(cred.tecido, cred.arquivo)]
keys = np.array([str(p) for p in paths])
if a.cache.exists() and np.array_equal(np.load(a.cache)["keys"], keys):
    c = np.load(a.cache)
    Q, D = c["q"], c["d"]
else:
    Q, D = feats(paths)
    np.savez(a.cache, keys=keys, q=Q, d=D)

cur = a.curadoria
if cur.exists():  # picked by eye: H&E (or close), normal tissue, a real micrograph, the right organ
    pick = pd.read_csv(cur)
    m = cred.merge(pick.rename(columns={"tecido": "novo", "pasta": "tecido"}), on=["tecido", "arquivo"], how="left")
    keep = m.novo.notna().values
    cred = cred.assign(tecido=m.novo.values)
    print(f"\n{len(cred)} commons images, {keep.sum()} kept by hand ({cur})")
else:
    keep = model.prompt_probs(Q, KIND).argmax(1) == 0
    print(f"\n{len(cred)} commons images, {keep.sum()} look like H&E sections, the rest are drawings, IHC, etc.")
cred, Q, D = cred[keep].reset_index(drop=True), Q[keep], D[keep]
counts = cred.tecido.value_counts()
ok = cred.tecido.map(counts).values >= 5  # fewer images than that cannot be both learned and tested
cred, Q, D = cred[ok].reset_index(drop=True), Q[ok], D[ok]
y = np.array([NAMES.index(t) for t in cred.tecido])
groups = cred.autor.fillna("").where(cred.autor.fillna("") != "", cred.titulo).str.lower().str[:40].values
print(f"{len(cred)} images, {len(set(y))} tissues, {len(set(groups))} authors")

X = np.concatenate([Q, D], 1)
pred = {"zero-shot (sem treino)": model.zero_shot(emb=Q).argmax(1)}
for name in ["vizinho mais parecido (DINOv2)", "vizinho mais parecido (QuiltNet)", "classificador (QuiltNet + DINOv2)"]:
    pred[name] = np.zeros_like(y)
for tr, te in GroupKFold(a.folds).split(X, y, groups):
    pred["vizinho mais parecido (DINOv2)"][te] = y[tr][(D[te] @ D[tr].T).argmax(1)]
    pred["vizinho mais parecido (QuiltNet)"][te] = y[tr][(Q[te] @ Q[tr].T).argmax(1)]
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
    pred["classificador (QuiltNet + DINOv2)"][te] = clf.classes_[clf.predict_proba(X[te]).argmax(1)]

rows = [{"metodo": k, "acertos": int((p == y).sum()), "n": len(y), "acuracia": (p == y).mean(),
         "acuracia_balanceada": balanced_accuracy_score(y, p), "acaso": 1 / len(set(y))} for k, p in pred.items()]
best = pred["classificador (QuiltNet + DINOv2)"]
per = pd.DataFrame({"tecido": [NAMES[i] for i in y], "certo": best == y}).groupby("tecido").certo.agg(["sum", "count", "mean"])

clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X, y)
Path("models").mkdir(exist_ok=True)
np.savez("models/tissue_clf.npz", coef=clf.coef_, intercept=clf.intercept_, classes=np.array([NAMES[i] for i in clf.classes_]))
if lab:
    lq, ld = feats(lab)
    truth = [p.parent.name for p in lab]
    guess = [NAMES[i] for i in clf.classes_[clf.predict_proba(np.concatenate([lq, ld], 1)).argmax(1)]]
    print("\nlab photos (never seen, other slides, through the eyepiece):")
    for p, t, g in zip(lab, truth, guess):
        print(f"  {p.name:20s} {t:20s} -> {g}{'' if t == g else '   X'}")
    hit = sum(t == g for t, g in zip(truth, guess))
    rows.append({"metodo": "classificador, fotos do laboratorio", "acertos": hit, "n": len(lab), "acuracia": hit / len(lab),
                 "acuracia_balanceada": balanced_accuracy_score(truth, guess), "acaso": 1 / len(set(y))})

res = pd.DataFrame(rows)
print("\n" + res.to_string(index=False))
print("\n" + per.sort_values("mean").to_string())
res.to_csv("results/tecidos_eval.csv", index=False)
per.to_csv("results/tecidos_por_tecido.csv")
