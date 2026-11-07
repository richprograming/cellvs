"""docs/banner.jpg: GTEx crops (CC BY 4.0) with their tissue, for the README.
examples are picked: for each tissue, the 1 mm crop the final model is most sure about and gets right,
so the banner shows typical-looking tissue. no percentages: these slides are in the final model's training set,
and the honest hit rates are in results/.

python scripts/readme_banner.py      # needs data/gtex, data/emb_cache.npz and models/tissue_clf.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cellvs.histo import Model, clf_logp

TISSUES = ["tireoide", "cerebelo", "baco", "pulmao", "rim", "testiculo",
           "figado", "pancreas", "esofago", "colon", "musculo esqueletico", "tecido adiposo"]
LABEL = {"tireoide": "tireoide", "cerebelo": "cerebelo", "baco": "baço", "pulmao": "pulmão", "rim": "rim",
         "testiculo": "testículo", "figado": "fígado", "pancreas": "pâncreas", "esofago": "esôfago", "colon": "cólon",
         "musculo esqueletico": "músculo esquelético", "tecido adiposo": "tecido adiposo"}
SIDE, COLS, PAD, STRIP = 230, 6, 6, 40
FONT_B = "C:/Windows/Fonts/segoeuib.ttf" if Path("C:/Windows/Fonts/segoeuib.ttf").exists() else "DejaVuSans-Bold.ttf"

gt = pd.read_csv("data/gtex/creditos.csv")
gt = gt[gt.arquivo.str.contains("_1mm_") & gt.tecido.isin(TISSUES)].reset_index(drop=True)
gt["path"] = [str(Path("data/gtex") / t / f) for t, f in zip(gt.tecido, gt.arquivo)]
cache = np.load("data/emb_cache.npz", allow_pickle=True)["d"].item()
gt = gt[gt.path.isin(cache)].reset_index(drop=True)
E = np.stack([cache[p] for p in gt.path])

clf = dict(np.load("models/tissue_clf.npz"))
names = [str(n) for n in clf["names"]]
s = Model().zero_shot(E, names) + clf_logp(clf, E)
p = np.exp(s - s.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)
gt["guess"] = [names[i] for i in p.argmax(1)]
gt["prob"] = p.max(1)

rows = -(-len(TISSUES) // COLS)
W = COLS * SIDE + (COLS + 1) * PAD
H = rows * (SIDE + STRIP) + (rows + 1) * PAD
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)
f_name = ImageFont.truetype(FONT_B, 17)
for k, t in enumerate(TISSUES):
    ok = gt[(gt.tecido == t) & (gt.guess == t)].sort_values("prob", ascending=False)
    r = ok.iloc[0]
    x, y = PAD + (k % COLS) * (SIDE + PAD), PAD + (k // COLS) * (SIDE + STRIP + PAD)
    img.paste(Image.open(r.path).convert("RGB").resize((SIDE, SIDE), Image.LANCZOS), (x, y))
    draw.text((x + 2, y + SIDE + 8), LABEL[t], font=f_name, fill="#1f1f1d")
    print(f"{t:20s} {r.slide}  {r.prob:.2f}")
Path("docs").mkdir(exist_ok=True)
img.save("docs/banner.jpg", quality=88)
print("docs/banner.jpg", img.size)
