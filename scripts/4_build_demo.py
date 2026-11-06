"""build demo/index.html, a single file: real images + nearest neighbors + results

python scripts/4_build_demo.py
the page embeds BBBC021 images, so demo/ stays out of git
"""
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cellvs.evaluate import aggregate, center_on_dmso, l2norm, matches

CROP, SIDE = 640, 320
RAW = ROOT / "data/raw"
meta = pd.read_csv(ROOT / "data/metadata.csv")
embs = {b: np.load(ROOT / f"results/emb_{b}.npy") for b in ["resnet50", "dinov2_vits14"]}


def channel(path):
    x = cv2.imread(str(RAW / path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    lo, hi = np.percentile(x, [1, 99.5])  # same stretch the models get
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    top, left = (x.shape[0] - CROP) // 2, (x.shape[1] - CROP) // 2
    return cv2.resize(x[top : top + CROP, left : left + CROP], (SIDE, SIDE), interpolation=cv2.INTER_AREA)


def field_uri(row):
    bgr = np.stack([channel(row.path_dapi), channel(row.path_tubulin), channel(row.path_actin)], -1)
    ok, buf = cv2.imencode(".jpg", (bgr * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def medoid_uri(rows):
    # the field closest to the group's mean embedding, so the picture is typical and not hand-picked
    e = l2norm(embs["dinov2_vits14"][rows.index])
    return field_uri(rows.iloc[int(np.argmax(e @ l2norm(e.mean(0, keepdims=True)).T))])


def png_uri(name):
    return "data:image/png;base64," + base64.b64encode((ROOT / "results" / name).read_bytes()).decode()


def treatments(emb):
    e, t = aggregate(emb, meta)
    keep = ((t.compound != "DMSO") & t.moa.notna()).values
    return e[keep], t[keep].reset_index(drop=True)


_, t = treatments(embs["resnet50"])
comp, moa, batch = t.compound.values, t.moa.values, t.batch.values

nn, tables, per_class = {}, {}, {}
for b, raw in embs.items():
    for corr, emb in [("raw", raw), ("centered", center_on_dmso(raw, meta))]:
        e, tt = treatments(emb)
        assert tt.equals(t)
        sim = l2norm(e) @ l2norm(e).T
        res = json.loads((ROOT / f"results/emb_{b}{'_raw' if corr == 'raw' else ''}.json").read_text())
        for rule, excl in [("nsc", False), ("nscb", True)]:
            m = matches(e, t, excl)
            assert m.hit.mean() == res[rule] and len(m) == res[f"{rule}_n"], "demo disagrees with the results table"
            evaluated = dict(zip(m.i, m.j))
            lists = []
            for i in range(len(t)):
                if i not in evaluated:
                    lists.append(None)
                    continue
                pool = comp != comp[i]
                if excl:
                    pool = pool & (batch != batch[i])
                idx = np.where(pool)[0]
                top = idx[np.argsort(-sim[i, idx], kind="stable")[:5]]
                assert top[0] == evaluated[i]
                lists.append([[int(j), round(float(sim[i, j]), 4)] for j in top])
            nn[f"{b}|{corr}|{rule}"] = lists
            tables[f"{b}|{corr}|{rule}"] = {"hits": int(m.hit.sum()), "n": len(m), "value": res[rule], "chance": res[f"{rule}_chance"]}
            g = m.groupby("moa").hit
            per_class[f"{b}|{corr}|{rule}"] = {k: [int(v.sum()), len(v)] for k, v in g}

print("building images...", flush=True)
images = []
for _, row in t.iterrows():
    rows = meta[(meta.compound == row.compound) & (meta.conc == row.conc) & (meta.batch == row.batch)]
    images.append(medoid_uri(rows))
dmso = {w: medoid_uri(meta[(meta.compound == "DMSO") & (meta.batch == w)]) for w in meta.batch.unique()}

checks = pd.read_csv(ROOT / "results/leak_checks.csv")
data = {
    "treatments": t[["compound", "conc", "batch", "moa"]].to_dict("records"),
    "images": images,
    "dmso": dmso,
    "nn": nn,
    "tables": tables,
    "perClass": per_class,
    "moas": sorted(t.moa.unique()),
    "checks": checks.to_dict("records"),
    "leak": json.loads((ROOT / "results/leak_checks_summary.json").read_text()),
    "batch": pd.read_csv(ROOT / "results/batch_vs_moa.csv").to_dict("records"),
    "counts": {
        "images": len(meta), "labeled": int(meta.moa.notna().sum()), "dmso": int(meta.moa.isna().sum()),
        "compounds": int(meta[meta.moa.notna()].compound.nunique()), "moas": int(meta.moa.nunique()),
        "weeks": int(meta.batch.nunique()), "plates": int(meta.plate.nunique()), "treatments": len(t),
    },
    "umap": {"moa": png_uri("umap_dinov2_vits14_raw_moa.png"), "batch": png_uri("umap_dinov2_vits14_raw_batch.png")},
}

tissue = None
if (ROOT / "results/tecidos_eval.csv").exists():
    from cellvs.histo import read_photo

    def photo_uri(path, side=360):
        rgb = read_photo(path)
        s = side / max(rgb.shape[:2])
        bgr = cv2.cvtColor(cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    ext = ROOT / "data/tecidos_teste_externo.csv"  # the lab's photos: only in this local page, never in git
    lab = pd.read_csv(ext) if ext.exists() else pd.DataFrame()
    tissue = {
        "eval": pd.read_csv(ROOT / "results/tecidos_eval.csv").to_dict("records"),
        "per": pd.read_csv(ROOT / "results/tecidos_por_tecido.csv").to_dict("records"),
        "lab": [{"img": photo_uri(ROOT / r.caminho), "tecido": r.tecido, "top": [[r.top1, r.p1], [r.top2, r.p2], [r.top3, r.p3]]}
                for r in lab.itertuples()],
    }
data["tissue"] = tissue

template = (ROOT / "scripts/demo_template.html").read_text(encoding="utf-8")
page = template.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
out = ROOT / "demo/index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(page, encoding="utf-8")
print(out, round(out.stat().st_size / 2**20, 2), "MiB")
