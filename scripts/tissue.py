"""suggest the tissue of microscope photos taken through the eyepiece

python scripts/tissue.py --input fotos/                 # every photo in the folder
python scripts/tissue.py --input fotos/ --watch         # keeps going, answers each photo saved in the folder

two answers per photo:
- zero-shot: QuiltNet (pathology CLIP) compares the photo with a sentence per tissue, needs no examples
- referencia: the most similar photo (DINOv2) in data/tecidos/<tecido>/, photos you already know. copy a photo
  there once you are sure of its tissue and the next ones get compared with it
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellvs.histo import TISSUES, Model, read_photo

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
NAMES = list(TISSUES)

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--ref", type=Path, default=Path("data/tecidos"))
ap.add_argument("--out", type=Path, default=Path("data/tecidos_respostas.csv"))
ap.add_argument("--ref-min", type=float, default=0.8, help="reference similarity needed to trust it over the zero-shot")
ap.add_argument("--zs-min", type=float, default=0.2, help="zero-shot probability below which the answer is 'incerto'")
ap.add_argument("--watch", action="store_true")
a = ap.parse_args()


def images_in(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS) if folder.exists() else []


model = Model()
refs = images_in(a.ref)
ref_e = model.similarity_emb([read_photo(p) for p in refs]) if refs else None
ref_t = [p.parent.name for p in refs]
print(f"{len(refs)} reference photos, {len(set(ref_t))} tissues", flush=True)


def answer(paths):
    rgbs = [read_photo(p) for p in paths]
    prob = model.zero_shot(rgbs)
    emb = model.similarity_emb(rgbs) if refs else [None] * len(paths)
    rows = []
    for p, e, pr in zip(paths, emb, prob):
        top = np.argsort(-pr)[:3]
        zs = f"{NAMES[top[0]]}" if pr[top[0]] >= a.zs_min else "incerto"
        row = {"imagem": p.name, "zero_shot": "; ".join(f"{NAMES[i]} {pr[i]:.2f}" for i in top)}
        sims = np.array([])
        if refs:
            sims = ref_e @ e
            sims[sims > 0.999] = -np.inf  # the same photo, copied into the reference folder
        if len(sims) and np.isfinite(sims.max()):
            j = int(np.argmax(sims))
            row["referencia"] = f"{ref_t[j]} ({refs[j].name}) {sims[j]:.3f}"
            row["sugestao"] = ref_t[j] if sims[j] >= a.ref_min else zs
        else:
            row["referencia"] = ""
            row["sugestao"] = zs
        rows.append(row)
        print(f"{p.name}: {row['sugestao']}  | zero-shot: {row['zero_shot']}  | ref: {row['referencia'] or '-'}", flush=True)
    a.out.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(a.out, mode="a", header=not a.out.exists(), index=False)


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
