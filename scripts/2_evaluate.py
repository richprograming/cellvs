import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/ is what python puts on sys.path, not the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellvs.evaluate import aggregate, center_on_dmso, matches, nsc, nscb, per_moa

ap = argparse.ArgumentParser()
ap.add_argument("--meta", type=Path, default=Path("data/metadata.csv"))
ap.add_argument("--emb", type=Path, required=True)
ap.add_argument("--no-center", action="store_true", help="skip DMSO centering")
a = ap.parse_args()

meta = pd.read_csv(a.meta)
emb = np.load(a.emb)
assert len(emb) == len(meta), "embeddings and metadata out of sync"

if not a.no_center:
    emb = center_on_dmso(emb, meta)

# controls out, labeled treatments only
emb, tmeta = aggregate(emb, meta)
keep = (tmeta.compound != "DMSO") & tmeta.moa.notna()
emb, tmeta = emb[keep.values], tmeta[keep].reset_index(drop=True)

res = {
    "emb": a.emb.name,
    "centered": not a.no_center,
    "n_treatments": len(tmeta),
}
res["nsc"], res["nsc_n"] = nsc(emb, tmeta)
res["nscb"], res["nscb_n"] = nscb(emb, tmeta)
res["nsc_chance"] = float(matches(emb, tmeta, False).chance.mean())
res["nscb_chance"] = float(matches(emb, tmeta, True).chance.mean())

print(json.dumps(res, indent=2))
tag = a.emb.stem + ("_raw" if a.no_center else "")
Path("results").mkdir(exist_ok=True)
Path(f"results/{tag}.json").write_text(json.dumps(res, indent=2))
per_moa(emb, tmeta).to_csv(f"results/{tag}_per_moa.csv")
