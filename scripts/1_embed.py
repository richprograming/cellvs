"""extract embeddings, one forward pass per channel, then concat.

python scripts/1_embed.py --meta data/metadata.csv --img-root data/raw --backbone resnet50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellvs.embed import CHANNELS, embed, get_backbone, read

ap = argparse.ArgumentParser()
ap.add_argument("--meta", type=Path, default=Path("data/metadata.csv"))
ap.add_argument("--img-root", type=Path, required=True)
ap.add_argument("--backbone", default="resnet50")
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()

out = a.out or Path(f"results/emb_{a.backbone}.npy")
meta = pd.read_csv(a.meta)
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = get_backbone(a.backbone).to(dev)

feats = []
for s in range(0, len(meta), a.bs):
    chunk = meta.iloc[s : s + a.bs]
    fields = [[read(a.img_root / row[f"path_{ch}"]) for ch in CHANNELS] for _, row in chunk.iterrows()]
    feats.append(embed(net, fields, a.size, dev))
    if (s // a.bs) % 20 == 0:
        print(f"{s}/{len(meta)}", flush=True)

out.parent.mkdir(exist_ok=True)
np.save(out, np.concatenate(feats))
print("saved", out)
