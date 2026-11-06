"""download public histology images from Wikimedia Commons, one folder per tissue, with a credits csv

python scripts/fetch_commons.py --out data/commons
the images keep their own licenses (CC BY / CC BY-SA / public domain): see creditos.csv before reusing them
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellvs.histo import TISSUES

EXTRA = {
    "ducto deferente": ["vas deferens histology"], "colon": ["large intestine histology"],
    "musculo cardiaco": ["cardiac muscle histology", "myocardium histology"], "nervo": ["nerve fascicle histology"],
    "tecido adiposo": ["fat tissue histology"], "linfonodo": ["lymph node H&E"], "baco": ["splenic red pulp"],
    "intestino delgado": ["ileum histology", "jejunum histology", "duodenum histology"], "bexiga": ["bladder histology"],
    "osso": ["compact bone histology", "osteon micrograph"], "cartilagem": ["hyaline cartilage histology"],
    "pele": ["epidermis histology"], "rim": ["renal cortex histology"], "pulmao": ["alveoli histology"],
}
SKIP = re.compile(r"drawing|diagram|description|compendium|cajal|golgi|silver|illustrat|scheme|schema|sketch|"
                  r"\(1[89]\d\d\)|plate|figure|fig\.|anatomy of|gray\d|mri|ct scan|x-ray|gross|labeled|labelled|"
                  r"cancer|carcinoma|tumou?r|adenoma|metasta|lymphoma|glioma|medulloblastoma|sarcoma|melanoma|"
                  r"infarct|fibrosis|cirrhosis|hepatitis|itis\b|disease|syndrome|lesion|necrosis|ultrasound|"
                  r"electron|\bsem\b|\btem\b|immunohisto|\bihc\b|fluoresc", re.I)
API = "https://commons.wikimedia.org/w/api.php"

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("data/commons"))
ap.add_argument("--per-tissue", type=int, default=40)
a = ap.parse_args()

s = requests.Session()
s.headers["User-Agent"] = "cellvs-histology/0.1 (student research project)"


def api(**params):
    for wait in [0, 5, 20, 60]:
        time.sleep(wait)
        r = s.get(API, params={"action": "query", "format": "json", **params})
        if r.ok and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
    raise RuntimeError(f"commons api answered {r.status_code} {r.text[:200]}")


def download(url, path):
    for wait in [0.3, 5, 30]:  # 429 comes back as an html page
        time.sleep(wait)
        r = s.get(url)
        if r.ok and r.headers.get("content-type", "").startswith("image/"):
            path.write_bytes(r.content)
            return True
    return False


def infos(titles, width):
    r = api(titles="|".join(titles), prop="imageinfo", iiprop="url|mime|extmetadata|size", iiurlwidth=width)
    return [(p["title"], p["pageid"], (p.get("imageinfo") or [{}])[0]) for p in r["query"]["pages"].values() if "pageid" in p]


credits = []
for tissue, en in TISSUES.items():
    titles = []
    for q in [f"{en} histology", f"{en} H&E", f"normal {en} micrograph"] + EXTRA.get(tissue, []):
        r = api(list="search", srnamespace=6, srsearch=q + " filetype:bitmap", srlimit=100)
        titles += [h["title"] for h in r["query"]["search"] if not SKIP.search(h["title"]) and h["title"] not in titles]
    folder = a.out / tissue
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(0, len(titles), 40):
        if n >= a.per_tissue:
            break
        batch = infos(titles[i : i + 40], 960)
        small = [t for t, _, inf in batch if 500 <= inf.get("width", 0) < 960]
        # only standard thumbnail widths are served without rate limits: 960 for big images, 500 for the rest
        batch = [b for b in batch if b[2].get("width", 0) >= 960] + (infos(small, 500) if small else [])
        for title, pageid, info in batch:
            if n >= a.per_tissue or info.get("mime") not in ("image/jpeg", "image/png"):
                continue
            meta = info.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if not re.search(r"CC BY|CC0|Public domain|PD", lic, re.I) or "px-" not in info.get("thumburl", ""):
                continue
            path = folder / f"{pageid}.jpg"
            if not path.exists() and not download(info["thumburl"], path):
                print("  failed:", title, flush=True)
                continue
            artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
            credits.append({"tecido": tissue, "arquivo": path.name, "titulo": title, "autor": artist,
                            "licenca": lic, "url": info["descriptionurl"]})
            n += 1
    print(tissue, n, "images", flush=True)
    pd.DataFrame(credits).to_csv(a.out / "creditos.csv", index=False)
