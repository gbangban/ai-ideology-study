#!/usr/bin/env python3
"""Generate capability shift summary figure (Figure 3).

Horizontal diverging bar chart showing all benchmark deltas in one view:
large gains (Corr2Cause), large losses (EconCausal), and near-zero
changes (MMLU, HumanEval, GPQA, IFEval).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Publication defaults ---
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "axes.grid.axis": "x",
    "lines.linewidth": 1.5,
})

# Color coding by magnitude
LARGE_POS = "#009E73"      # green — large improvement
LARGE_NEG = "#E76F51"      # burnt coral — large regression
NEAR_ZERO = "#B0BEC5"      # cool gray — within noise

benchmarks = [
    "Corr2Cause",
    "EconCausal T1 Econ",
    "EconCausal T1 Fin",
    "EconCausal T3",
    "EconCausal T2",
    "GPQA Diamond",
    "IFEval (loose)",
    "IFEval (strict)",
    "MMLU Overall",
    "MMLU Humanities",
    "MMLU Social Sci",
    "MMLU STEM",
    "HumanEval",
]
deltas = [
    +38.3,   # Corr2Cause
    -12.36,  # T1 Econ
    -13.49,  # T1 Fin
    -10.80,  # T3
    -3.87,   # T2
    -1.51,   # GPQA
    -1.75,   # IFEval loose
    -1.20,   # IFEval strict
    -0.76,   # MMLU Overall
    -0.83,   # MMLU Humanities
    -0.49,   # MMLU Social Sci
    -0.29,   # MMLU STEM
    0.00,    # HumanEval
]

colors = []
for d in deltas:
    if d >= 5:
        colors.append(LARGE_POS)
    elif d <= -3:
        colors.append(LARGE_NEG)
    else:
        colors.append(NEAR_ZERO)

fig, ax = plt.subplots(figsize=(5.5, 4.2))

y = np.arange(len(benchmarks))
bars = ax.barh(y, deltas, color=colors, height=0.6,
               edgecolor="white", linewidth=0.5, zorder=3)

# Zero reference line
ax.axvline(x=0, color="#333", linewidth=0.8, zorder=2, linestyle="-")

# Value labels
for bar, d in zip(bars, deltas):
    h = bar.get_width()
    if abs(h) > 0.5:
        ha = "left" if h > 0 else "right"
        offset = 0.3 if h > 0 else -0.3
        ax.annotate(f"{d:+.1f}pp",
                    xy=(h, bar.get_y() + bar.get_height() / 2),
                    xytext=(offset, 0), textcoords="offset points",
                    ha=ha, va="center", fontsize=7.5, color="#333")

ax.set_yticks(y)
ax.set_yticklabels(benchmarks, fontsize=7.5)
ax.set_xlabel("Accuracy change (percentage points)")
ax.set_xlim(-16, 42)
ax.set_title("SFT capability shifts across benchmarks", pad=8)

# Subtle group separators
for sep_y in [3.5, 6.5, 10.5]:
    ax.axhline(y=sep_y, color="#CCC", linewidth=0.5, linestyle="--", zorder=1)

# Group labels on right
ax.text(43, 1.5, "Formal logic", fontsize=6.5, va="center",
        color=LARGE_POS, fontweight="bold", style="italic")
ax.text(43, 5.5, "Applied ID", fontsize=6.5, va="center",
        color=LARGE_NEG, fontweight="bold", style="italic")
ax.text(43, 9.5, "General", fontsize=6.5, va="center",
        color=NEAR_ZERO, fontweight="bold", style="italic")

fig.savefig("paper/iclr2026/figures/fig_capability_shift.pdf")
fig.savefig("paper/iclr2026/figures/fig_capability_shift.png", dpi=300)
print("Saved: fig_capability_shift.pdf, fig_capability_shift.png")
