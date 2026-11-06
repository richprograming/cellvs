import numpy as np
import pandas as pd
import torch

from cellvs.embed import embed, get_backbone
from cellvs.library import build, threshold, top_k


class MoAPredictor:
    """nearest BBBC021 treatments of a new field; "desconhecido" below the calibrated similarity.
    no dmso correction: new images may come without controls, and it did not help here (see notes.md)"""

    def __init__(self, meta="data/metadata.csv", backbone="dinov2_vits14", hold_out_week=None, size=224):
        meta = pd.read_csv(meta)
        raw = np.load(f"results/emb_{backbone}.npy")
        in_lib = (meta.batch != hold_out_week).values
        self.lib_e, self.lib_t = build(raw[in_lib], meta[in_lib].reset_index(drop=True))
        labeled = in_lib & meta.moa.notna().values
        self.thr = threshold(self.lib_e, self.lib_t, raw[labeled], meta[labeled])
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = get_backbone(backbone).to(self.dev)
        self.size = size

    def predict(self, fields, k=5):
        """fields: list of [dapi, tubulin, actin] arrays -> one dict per field"""
        idx, sims = top_k(embed(self.net, fields, self.size, self.dev), self.lib_e, k)
        out = []
        for ii, ss in zip(idx, sims):
            t = self.lib_t
            out.append({
                "moa": t.moa[ii[0]] if ss[0] >= self.thr else "desconhecido",
                "similaridade": float(ss[0]),
                "limite": self.thr,
                "vizinhos": [{"composto": t.compound[i], "conc": float(t.conc[i]), "semana": t.batch[i],
                              "moa": t.moa[i], "similaridade": float(s)} for i, s in zip(ii, ss)],
            })
        return out
