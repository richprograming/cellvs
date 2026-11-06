"""suggest a moa for new images by looking up the nearest treatments of the BBBC021 library

python scripts/predict.py --input new.csv                 # csv with path_dapi, path_tubulin, path_actin
python scripts/predict.py --input folder/                 # one 3-channel image per field: blue=dna, green=tubulin, red=actin
python scripts/predict.py --input folder/ --watch         # keeps going, answers each image saved in the folder

only meaningful for images made like BBBC021: MCF-7, 24 h with a compound, same 3 stains, same magnification
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellvs.embed import CHANNELS, embed, get_backbone, read, read_rgb
from cellvs.library import build, threshold, top_k

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--img-root", type=Path, default=Path("data/raw"), help="root for the paths in a csv input")
ap.add_argument("--backbone", default="dinov2_vits14")
ap.add_argument("--meta", type=Path, default=Path("data/metadata.csv"))
ap.add_argument("--out", type=Path, default=Path("data/predictions.csv"))
ap.add_argument("--watch", action="store_true")
ap.add_argument("--hold-out-week", help="leave this week out of the library, to test on images it has never seen")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()

# no dmso correction: new images may come without controls, and it did not help here (see notes.md)
meta = pd.read_csv(a.meta)
raw = np.load(f"results/emb_{a.backbone}.npy")
in_lib = (meta.batch != a.hold_out_week).values
lib_e, lib_t = build(raw[in_lib], meta[in_lib].reset_index(drop=True))
labeled = in_lib & meta.moa.notna().values
thr = threshold(lib_e, lib_t, raw[labeled], meta[labeled])
print(f"library: {len(lib_t)} treatments, unknown below similarity {thr:.4f}", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = get_backbone(a.backbone).to(dev)


def answer(names, fields):
    idx, sims = top_k(embed(net, fields, a.size, dev), lib_e, a.k)
    rows = []
    for name, ii, ss in zip(names, idx, sims):
        moa = lib_t.moa[ii[0]] if ss[0] >= thr else "desconhecido"
        top = "; ".join(f"{lib_t.compound[i]} {lib_t.conc[i]:g} {lib_t.batch[i]} ({lib_t.moa[i]}) {s:.4f}" for i, s in zip(ii, ss))
        rows.append({"imagem": name, "moa_sugerido": moa, "similaridade": round(float(ss[0]), 4), "limite": round(thr, 4), "vizinhos": top})
        print(f"{name}: {moa} ({ss[0]:.4f})", flush=True)
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(a.out, mode="a", header=not a.out.exists(), index=False)


def images_in(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


if a.input.suffix.lower() == ".csv":
    items = pd.read_csv(a.input)
    for s in range(0, len(items), a.bs):
        chunk = items.iloc[s : s + a.bs]
        fields = [[read(a.img_root / row[f"path_{ch}"]) for ch in CHANNELS] for _, row in chunk.iterrows()]
        answer(chunk.path_dapi.tolist(), fields)
elif not a.watch:
    paths = images_in(a.input)
    for s in range(0, len(paths), a.bs):
        answer([p.name for p in paths[s : s + a.bs]], [read_rgb(p) for p in paths[s : s + a.bs]])
else:
    print("watching", a.input, "(ctrl+c to stop)", flush=True)
    done, sizes = set(), {}
    while True:
        for p in images_in(a.input):
            if p in done:
                continue
            size = p.stat().st_size
            if sizes.get(p) != size:  # still being written: wait until the size stops changing
                sizes[p] = size
                continue
            try:
                answer([p.name], [read_rgb(p)])
            except (ValueError, FileNotFoundError) as e:
                print("skipped:", e, flush=True)
            done.add(p)
        time.sleep(1)
