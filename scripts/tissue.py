"""suggest the tissue of histology photos, including phone photos taken through the eyepiece

python scripts/tissue.py --input fotos/                 # every photo in the folder
python scripts/tissue.py --input foto.jpg
python scripts/tissue.py --input fotos/ --watch         # keeps going, answers each photo saved in the folder

uses models/tissue_clf.npz (scripts/tissue_train.py) when it exists, otherwise zero-shot only over all tissues.
prints the 3 most likely tissues: the first one is right about half the time, see results/tecidos_eval.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellvs.histo import TissuePredictor, read_photo

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--clf", type=Path, default=Path("models/tissue_clf.npz"))
ap.add_argument("--out", type=Path, default=Path("data/tecidos_respostas.csv"))
ap.add_argument("--watch", action="store_true")
a = ap.parse_args()

pred = TissuePredictor(a.clf)
names = pred.names
print(f"{len(names)} tissues, {'zero-shot + classifier' if pred.clf else 'zero-shot only (no ' + str(a.clf) + ')'}", flush=True)


def answer(paths):
    prob = pred.probs([read_photo(p) for p in paths])
    rows = []
    for p, pr in zip(paths, prob):
        top = np.argsort(-pr)[:3]
        txt = ", ".join(f"{names[i]} {pr[i]:.0%}" for i in top)
        rows.append({"imagem": p.name, "tecido": names[top[0]], "top3": txt})
        print(f"{p.name}: {txt}", flush=True)
    a.out.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(a.out, mode="a", header=not a.out.exists(), index=False)


def images_in(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)


if not a.watch:
    paths = images_in(a.input) if a.input.is_dir() else [a.input]
    for s in range(0, len(paths), 16):
        answer(paths[s : s + 16])
else:
    print("watching", a.input, "(ctrl+c to stop)", flush=True)
    done, sizes = set(), {}
    while True:
        for p in images_in(a.input):
            if p in done:
                continue
            size = p.stat().st_size
            if sizes.get(p) != size:  # still being written
                sizes[p] = size
                continue
            try:
                answer([p])
            except (FileNotFoundError, ValueError, IndexError) as e:
                print("skipped:", p.name, e, flush=True)
            done.add(p)
        time.sleep(1)
