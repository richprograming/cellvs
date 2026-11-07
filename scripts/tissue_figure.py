"""results/tecidos_por_tecido.png: per-tissue hit rate, first guess vs any of the 3 guesses, one panel per test set"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e6e5e0"
TOP1, TOP3 = "#2a78d6", "#d95926"  # passes the categorical palette checks on white
FINAL = "B. zero-shot + classificador (Commons + GTEx)"
SUB = {"Commons": "Wikimedia Commons\nimagens de autores nunca vistos", "GTEx": "GTEx\nlâminas de doadores nunca vistos"}

per_all = pd.read_csv("results/tecidos_por_tecido.csv")
ev = pd.read_csv("results/tecidos_eval.csv")
tests = [t for t in SUB if t in set(per_all.teste)]
rows = max((per_all.teste == t).sum() for t in tests)

fig, axes = plt.subplots(1, len(tests), figsize=(4.6 * len(tests), 0.24 * rows + 1.9), dpi=150, squeeze=False)
for ax, test in zip(axes[0], tests):
    per = per_all[per_all.teste == test].sort_values(["top1", "top3"]).reset_index(drop=True)
    best = ev[(ev.teste == test) & (ev.metodo == FINAL)].iloc[0]
    for i, r in per.iterrows():
        ax.plot([r.top1, r.top3], [i, i], color=GRID, lw=2, solid_capstyle="round", zorder=1)
    ax.scatter(per.top3, per.index, s=20, color=TOP3, zorder=2, label="certo entre os 3 palpites")
    ax.scatter(per.top1, per.index, s=30, color=TOP1, edgecolor="white", linewidth=1, zorder=3, label="certo no 1º palpite")
    ax.set_yticks(per.index, [f"{t}  ({n})" for t, n in zip(per.tecido, per.n)], color=INK, fontsize=7.5)
    ax.set_ylim(-0.7, len(per) - 0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"{SUB[test]}\n{best.top1:.0%} no 1º palpite, {best.top3:.0%} entre 3, acaso {best.acaso_top1:.0%}",
                 loc="left", color=INK, fontsize=9)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower right", ncol=2, frameon=False, fontsize=7.5, labelcolor=INK)
fig.text(0.01, 0.012, "entre parênteses: número de imagens (no GTEx, recortes)\nmodelo treinado com Commons + GTEx",
         color=MUTED, fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=3)
Path("results").mkdir(exist_ok=True)
fig.savefig("results/tecidos_por_tecido.png", facecolor="white")
print("results/tecidos_por_tecido.png")
