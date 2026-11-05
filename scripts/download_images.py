"""download only the tifs listed in the metadata, using http range requests on the plate zips

python scripts/download_images.py --meta data/metadata.csv --raw data/raw
safe to re-run: files already on disk are skipped
"""
import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://data.broadinstitute.org/bbbc/BBBC021"
CHANNELS = ["dapi", "tubulin", "actin"]


class RemoteFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        self.http = requests.Session()  # keep-alive: a new tls handshake per read was ~2 s/file
        retry = Retry(total=8, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        self.http.mount("https://", HTTPAdapter(max_retries=retry))
        self.size = int(self.http.head(url, timeout=60).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        r = self.http.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        if r.status_code != 206:  # a 200 would silently hand back the start of the file
            raise IOError(f"no range support: {r.status_code}")
        b[: len(r.content)] = r.content
        self.pos += len(r.content)
        return len(r.content)


def fetch_plate(plate, plate_paths, raw):
    url = f"{BASE}/BBBC021_v1_images_{plate}.zip"
    with zipfile.ZipFile(io.BufferedReader(RemoteFile(url), 1 << 20)) as z:
        members = {Path(n).name: n for n in z.namelist()}
        for p in plate_paths:
            dest = raw / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(z.read(members[Path(p).name]))  # zipfile checks the crc
            tmp.replace(dest)
    print(plate, len(plate_paths), "files", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--meta", type=Path, default=Path("data/metadata.csv"))
ap.add_argument("--raw", type=Path, default=Path("data/raw"))
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()

meta = pd.read_csv(a.meta)
paths = pd.concat([meta[f"path_{ch}"] for ch in CHANNELS]).unique()
missing = [p for p in paths if not (a.raw / p).exists()]
print(len(paths), "files listed,", len(missing), "to download", flush=True)

by_plate = {}
for p in missing:
    by_plate.setdefault(p.split("/")[1], []).append(p)  # Week4/Week4_27481/x.tif

with ThreadPoolExecutor(a.workers) as ex:
    jobs = [ex.submit(fetch_plate, plate, ps, a.raw) for plate, ps in sorted(by_plate.items())]
    for job in jobs:
        job.result()
print("done")
