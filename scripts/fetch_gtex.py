"""crops of normal-tissue H&E slides from GTEx, via the NCI Imaging Data Commons (CC BY 4.0)

python scripts/fetch_gtex.py --gb 20        # ~600 slides, keeps only the crops (a few hundred MB)

- one slide per donor and tissue, picked at random with a fixed seed
- only the ~2 um/px level of each slide is downloaded (~30 MB instead of ~450 MB), then deleted
- crops of 2 mm and 1 mm, like a 10x and a 20x objective, where at least 60% of the crop is tissue
- the label is the GTEx sampling site: a crop can show some neighboring tissue (fat around a thyroid, say)
- resumable: slides already in data/gtex/creditos.csv are skipped
"""
import argparse
import io
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import requests
from idc_index import IDCClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellvs.histo import NAMES

# GTEx sampling site -> tissue name used in the project
SITES = {
    "Cerebellum": "cerebelo", "Cerebral cortex": "cortex cerebral", "Tibial nerve": "nervo",
    "Esophageal mucous membrane": "esofago", "Stomach": "estomago", "Ileum": "intestino delgado",
    "Transverse colon": "colon", "Liver": "figado", "Pancreas": "pancreas", "Renal cortex": "rim",
    "Urinary bladder": "bexiga", "Lung": "pulmao", "Heart": "musculo cardiaco", "Auricular appendage": "musculo cardiaco",
    "Skeletal muscle": "musculo esqueletico", "Skin of lower leg": "pele", "Skin of suprapubic region": "pele",
    "Spleen": "baco", "Thyroid": "tireoide", "Tibial artery": "arteria", "Aorta": "arteria", "Coronary artery": "arteria",
    "Adipose tissue": "tecido adiposo", "Omentum": "tecido adiposo", "Testis": "testiculo", "Uterus": "utero", "Ovary": "ovario",
}
assert set(SITES.values()) <= set(NAMES)
UM = 2.0  # wanted pixel spacing, micrometers
CROPS = {1000: 4, 500: 4}  # crop side in pixels at 2 um/px -> how many per slide
OUT_SIDE = 512

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("data/gtex"))
ap.add_argument("--gb", type=float, default=20, help="download budget; each slide level is ~30 MB")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

c = IDCClient()
c.fetch_index("sm_index")
c.fetch_index("sm_instance_index")
sites = ", ".join(f"'{s}'" for s in SITES)
df = c.sql_query(f"""
SELECT si.crdc_instance_uuid instance_uuid, i.crdc_series_uuid series_uuid, si.instance_size size, s.ContainerIdentifier slide, i.PatientID donor,
       s.primaryAnatomicStructure_CodeMeaning site, i.source_DOI doi, i.license_short_name license
FROM sm_instance_index si JOIN sm_index s USING(SeriesInstanceUID) JOIN index i USING(SeriesInstanceUID)
WHERE i.collection_id = 'gtex' AND s.primaryAnatomicStructure_CodeMeaning IN ({sites})
  AND ROUND(si.PixelSpacing_0 * 1000, 1) = {UM}
""")
df["tecido"] = df.site.map(SITES)

# same number of slides per tissue, spread over its sites, one per donor
per_tissue = int(a.gb * 1e9 / df["size"].median() / df.tecido.nunique())
rng = np.random.default_rng(a.seed)
pick = []
for t, g in df.groupby("tecido"):
    g = g.iloc[rng.permutation(len(g))].drop_duplicates("donor")
    g = g.assign(r=g.groupby("site").cumcount())  # alternate sites: 1st of each, then 2nd of each...
    pick.append(g.sort_values("r", kind="stable").head(per_tissue))
pick = pd.concat(pick).reset_index(drop=True)
print(f"{len(pick)} slides, {per_tissue} per tissue, {pick['size'].sum() / 1e9:.1f} GB to download")

a.out.mkdir(parents=True, exist_ok=True)
cred_path = a.out / "creditos.csv"
done = set(pd.read_csv(cred_path).slide) if cred_path.exists() else set()
pick = pick[~pick.slide.isin(done)]
print(f"{len(done)} already done, {len(pick)} to go")


def read_level(path):
    ds = pydicom.dcmread(path)
    if ds.get("DimensionOrganizationType") != "TILED_FULL":
        raise ValueError("not TILED_FULL")
    W, H, tw, th = ds.TotalPixelMatrixColumns, ds.TotalPixelMatrixRows, ds.Columns, ds.Rows
    nx = -(-W // tw)
    img = np.full((-(-H // th) * th, nx * tw, 3), 255, np.uint8)
    for k, f in enumerate(pydicom.encaps.generate_frames(ds.PixelData, number_of_frames=ds.NumberOfFrames)):
        y, x = divmod(k, nx)
        img[y * th : (y + 1) * th, x * tw : (x + 1) * tw] = np.array(Image.open(io.BytesIO(f)).convert("RGB"))
    return img[:H, :W]


def tissue_mask(rgb, f=8):
    """tissue = stained (saturated, not white) or textured; the glass is neither.
    fat is almost white, only its thin membranes give it away, so texture is measured at 4 um/px.
    holes enclosed by tissue (fat cells, alveoli) are filled"""
    size = (rgb.shape[1] // f, rgb.shape[0] // f)
    half = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(half, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0)
    tex = cv2.resize(np.abs(cv2.Laplacian(g, cv2.CV_32F)), size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)
    m = (((hsv[..., 1] > 25) & (hsv[..., 2] < 235)) | (tex > 4)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ff = np.pad(m, 1)
    cv2.floodFill(ff, None, (0, 0), 2)  # what the border reaches is glass, the rest is tissue
    return (ff[1:-1, 1:-1] != 2).astype(np.uint8), f


def crops(rgb, r):
    mask, f = tissue_mask(rgb)
    integ = cv2.integral(mask)
    out = []
    for side, n in CROPS.items():
        s = side // f
        H, W = mask.shape
        if H <= s or W <= s:
            continue
        frac = (integ[s:, s:] - integ[:-s, s:] - integ[s:, :-s] + integ[:-s, :-s]) / (s * s)
        ys, xs = np.where(frac[: H - s + 1, : W - s + 1] >= 0.6)
        got = []
        for i in r.permutation(len(ys)):
            y, x = ys[i], xs[i]
            if all(abs(y - gy) >= s or abs(x - gx) >= s for gy, gx in got):  # no overlap
                got.append((y, x))
                if len(got) == n:
                    break
        for y, x in got:
            c_ = rgb[y * f : y * f + side, x * f : x * f + side]
            out.append((side, cv2.resize(c_, (OUT_SIDE, OUT_SIDE), interpolation=cv2.INTER_AREA)))
    return out


def download(row, path):
    """IDC's public bucket keeps every instance at <series uuid>/<instance uuid>.dcm"""
    url = f"https://idc-open-data.s3.amazonaws.com/{row.series_uuid}/{row.instance_uuid}.dcm"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def work(row, seed):
    tmp = Path(tempfile.mkdtemp(prefix="gtex_"))
    try:
        download(row, tmp / "level.dcm")
        img = read_level(tmp / "level.dcm")
        folder = a.out / row.tecido
        folder.mkdir(exist_ok=True)
        files = []
        for k, (side, cr) in enumerate(crops(img, np.random.default_rng(seed))):
            name = f"{row.slide}_{side // 500}mm_{k}.jpg"
            cv2.imwrite(str(folder / name), cr[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
            files.append(name)
        return row, files, None
    except Exception as e:
        return row, [], repr(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


rows = []
with ThreadPoolExecutor(a.workers) as ex:
    futs = [ex.submit(work, r, a.seed + i) for i, r in enumerate(pick.itertuples())]
    for n, fut in enumerate(as_completed(futs), 1):
        r, files, err = fut.result()
        if err:
            print(f"  {r.slide}: {err}", flush=True)
            continue
        rows += [{"tecido": r.tecido, "arquivo": f, "slide": r.slide, "doador": r.donor, "local_gtex": r.site,
                  "licenca": r.license, "doi": r.doi} for f in files]
        if n % 20 == 0 or n == len(futs):
            new = pd.DataFrame(rows)
            new.to_csv(cred_path, mode="a", header=not cred_path.exists(), index=False)
            rows = []
            print(f"{n}/{len(futs)} slides", flush=True)
