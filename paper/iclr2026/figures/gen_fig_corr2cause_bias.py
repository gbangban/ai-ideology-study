#!/usr/bin/env python3
"""Generate Corr2Cause True-bias correction figure (Figure 1).

Shows baseline vs finetuned True-prediction rates per template type,
demonstrating how SFT corrects the pathological True-bias.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Publication defaults (ICLR single column: 5.5 in) ---
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "axes.grid.axis": "y",
    "lines.linewidth": 1.5,
})

# Okabe-Ito colorblind-safe palette
BASELINE_TRUE = "#56B4E9"    # sky blue
FINETUNED_TRUE = "#E76F51"   # burnt coral
GT_TRUE = "#8C8C8C"          # gray

templates = [
    "child",
    "non-child\ndescendant",
    "non-parent\nancestor",
    "has_confounder",
    "parent",
    "has_collider",
]
gt_true = [0.0, 5.7, 11.3, 8.8, 33.5, 33.1]
baseline_true = [19.1, 92.7, 96.9, 97.4, 43.8, 98.4]
finetuned_true = [0.0, 5.7, 12.7, 45.1, 33.5, 55.4]

fig, ax = plt.subplots(figsize=(5.5, 3.8))

x = np.arange(len(templates))
width = 0.22

# Ground truth True-rate (reference line)
ax.plot(x, gt_true, color=GT_TRUE, marker="x", markersize=6,
        linewidth=1.2, markeredgewidth=1.5, label="Ground truth True%",
        zorder=5, linestyle="--")

# Baseline True-prediction rate
bars1 = ax.bar(x - width, baseline_true, width, label="Baseline True%",
               color=BASELINE_TRUE, edgecolor="white", linewidth=0.5, zorder=3)

# Finetuned True-prediction rate
bars2 = ax.bar(x, finetuned_true, width, label="Finetuned True%",
               color=FINETUNED_TRUE, edgecolor="white", linewidth=0.5, zorder=3)

# Value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.0f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color="#333")

# GT labels
for xi, v in zip(x, gt_true):
    ax.annotate(f"{v:.1f}%",
                xy=(xi, v), xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=6, color=GT_TRUE, style="italic")

ax.set_xticks(x)
ax.set_xticklabels(templates, fontsize=8)
ax.set_ylabel("Predicted True%")
ax.set_ylim(0, 110)
ax.legend(loc="upper right", ncol=1)
ax.set_title("Corr2Cause: True-bias correction per template", pad=8)

fig.savefig("paper/iclr2026/figures/fig_corr2cause_bias.pdf")
fig.savefig("paper/iclr2026/figures/fig_corr2cause_bias.png", dpi=300)
print("Saved: fig_corr2cause_bias.pdf, fig_corr2cause_bias.png")
