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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellvs.embed import CHANNELS, read, read_rgb
from cellvs.moa import MoAPredictor

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

pred = MoAPredictor(a.meta, a.backbone, a.hold_out_week, a.size)
print(f"library: {len(pred.lib_t)} treatments, unknown below similarity {pred.thr:.4f}", flush=True)


def answer(names, fields):
    rows = []
    for name, r in zip(names, pred.predict(fields, a.k)):
        top = "; ".join(f"{v['composto']} {v['conc']:g} {v['semana']} ({v['moa']}) {v['similaridade']:.4f}" for v in r["vizinhos"])
        rows.append({"imagem": name, "moa_sugerido": r["moa"], "similaridade": round(r["similaridade"], 4),
                     "limite": round(r["limite"], 4), "vizinhos": top})
        print(f"{name}: {r['moa']} ({r['similaridade']:.4f})", flush=True)
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
