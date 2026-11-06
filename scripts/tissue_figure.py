"""results/tecidos_por_tecido.png: per-tissue hit rate, first guess vs any of the 3 guesses"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e6e5e0"
TOP1, TOP3 = "#2a78d6", "#d95926"  # passes the categorical palette checks on white

per = pd.read_csv("results/tecidos_por_tecido.csv").sort_values(["top1", "top3"]).reset_index(drop=True)
ev = pd.read_csv("results/tecidos_eval.csv")
best = ev[ev.metodo == "zero-shot + classificador"].iloc[0]

fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(per) + 1.4), dpi=150)
for i, r in per.iterrows():
    ax.plot([r.top1, r.top3], [i, i], color=GRID, lw=2, solid_capstyle="round", zorder=1)
ax.scatter(per.top3, per.index, s=22, color=TOP3, zorder=2, label="certo entre os 3 palpites")
ax.scatter(per.top1, per.index, s=34, color=TOP1, edgecolor="white", linewidth=1, zorder=3, label="certo no 1º palpite")
ax.set_yticks(per.index, [f"{t}  ({n})" for t, n in zip(per.tecido, per.n)], color=INK, fontsize=8)
ax.set_xlim(-0.02, 1.02)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=8)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_title(f"Acerto por tecido em imagens de autores nunca vistos\n"
             f"geral: {best.top1:.0%} no 1º palpite, {best.top3:.0%} entre 3  (acaso: {best.acaso_top1:.0%})",
             loc="left", color=INK, fontsize=10)
ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
fig.text(0.01, 0.005, "entre parênteses: número de imagens. fonte: Wikimedia Commons, curadoria manual",
         color=MUTED, fontsize=7)
fig.tight_layout()
Path("results").mkdir(exist_ok=True)
fig.savefig("results/tecidos_por_tecido.png", facecolor="white")
print("results/tecidos_por_tecido.png")
