"""extract embeddings, one forward pass per channel, then concat.

python scripts/1_embed.py --meta data/metadata.csv --img-root data/raw --backbone resnet50
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import models

CHANNELS = ["dapi", "tubulin", "actin"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_backbone(name):
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = torch.nn.Identity()
    elif name == "dinov2_vits14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    else:
        raise ValueError(name)
    return m.eval()


def load_channel(path, size):
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)  # 16 bit tif
    lo, hi = np.percentile(x, [1, 99.5])
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    x = np.stack([x] * 3, -1)  # fake rgb for imagenet weights
    return ((x - MEAN) / STD).transpose(2, 0, 1)


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
with torch.no_grad():
    for s in range(0, len(meta), a.bs):
        chunk = meta.iloc[s : s + a.bs]
        per_ch = []
        for ch in CHANNELS:
            x = np.stack([load_channel(a.img_root / p, a.size) for p in chunk[f"path_{ch}"]])
            per_ch.append(net(torch.from_numpy(x).to(dev)).cpu().numpy())
        feats.append(np.concatenate(per_ch, 1))
        if (s // a.bs) % 20 == 0:
            print(f"{s}/{len(meta)}", flush=True)

out.parent.mkdir(exist_ok=True)
np.save(out, np.concatenate(feats))
print("saved", out)
