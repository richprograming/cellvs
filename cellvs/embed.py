import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # the BBBC021 tifs have odd tags and libtiff warns on every read

import cv2
import numpy as np
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


def prep(x, size):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99.5])
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    x = np.stack([x] * 3, -1)  # fake rgb for imagenet weights
    # np.percentile returns float64 and numpy 2 promotes the whole array to it
    return ((x - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def read(path):
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(f"could not read {path}")
    return x


def read_rgb(path):
    """one file per field, channels as in a usual fluorescence composite: blue=dna, green=tubulin, red=actin"""
    x = read(path)
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError(f"{path}: expected an image with 3 color channels, got shape {x.shape}")
    return [x[..., 0], x[..., 1], x[..., 2]]  # opencv reads color as BGR


def embed(net, fields, size, dev):
    """fields: list of [dapi, tubulin, actin] arrays. one forward pass per channel, then concat"""
    per_ch = []
    with torch.no_grad():
        for c in range(len(CHANNELS)):
            x = np.stack([prep(f[c], size) for f in fields])
            per_ch.append(net(torch.from_numpy(x).to(dev)).cpu().numpy())
    return np.concatenate(per_ch, 1)
