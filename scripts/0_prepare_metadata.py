"""build data/metadata.csv from the original BBBC021 csv files

expects in --raw:  BBBC021_v1_image.csv  BBBC021_v1_moa.csv  (+ the image folders)
keeps labeled treatments and DMSO (needed as control for batch correction)
"""
import argparse
from pathlib import Path

import pandas as pd

# column names checked against the real BBBC021_v1_image.csv
C_COMPOUND = "Image_Metadata_Compound"
C_CONC = "Image_Metadata_Concentration"
C_PLATE = "Image_Metadata_Plate_DAPI"
C_WELL = "Image_Metadata_Well_DAPI"
CHANNELS = ["DAPI", "Tubulin", "Actin"]

ap = argparse.ArgumentParser()
ap.add_argument("--raw", type=Path, required=True)
ap.add_argument("--out", type=Path, default=Path("data/metadata.csv"))
a = ap.parse_args()

img = pd.read_csv(a.raw / "BBBC021_v1_image.csv")
moa = pd.read_csv(a.raw / "BBBC021_v1_moa.csv")
moa = moa.rename(columns={"compound": C_COMPOUND, "concentration": C_CONC})
moa = moa[moa.moa != "DMSO"]  # the csv lists DMSO as a 13th "moa"; it is the control, not a class

df = img.merge(moa, on=[C_COMPOUND, C_CONC], how="left")
df = df[df.moa.notna() | (df[C_COMPOUND] == "DMSO")]

out = pd.DataFrame({
    "compound": df[C_COMPOUND],
    "conc": df[C_CONC],
    "plate": df[C_PLATE],
    "well": df[C_WELL],
    "batch": df["Image_PathName_DAPI"].str.split("/").str[0],  # Week4/Week4_27481 -> Week4
    "moa": df.moa,
})
for ch in CHANNELS:
    out[f"path_{ch.lower()}"] = (
        df[f"Image_PathName_{ch}"].astype(str) + "/" + df[f"Image_FileName_{ch}"].astype(str)
    )

a.out.parent.mkdir(exist_ok=True, parents=True)
out.reset_index(drop=True).to_csv(a.out, index=False)
labeled = out[out.moa.notna()]
print(len(out), "images:", len(labeled), "labeled +", len(out) - len(labeled), "DMSO")
print(labeled.compound.nunique(), "compounds,", labeled.moa.nunique(), "moa classes,", out.batch.nunique(), "batches")
print(labeled[["compound", "conc"]].drop_duplicates().shape[0], "compound-concentrations")
