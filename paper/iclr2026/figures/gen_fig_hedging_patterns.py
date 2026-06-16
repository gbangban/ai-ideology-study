#!/usr/bin/env python3
"""Generate EconCausal hedging transitions figure (Figure 2).

Stacked bar chart showing the dominant regression patterns across
EconCausal tasks, highlighting +→mixed hedging as the primary artifact.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Publication defaults ---
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "axes.grid.axis": "y",
    "lines.linewidth": 1.5,
})

# Colors: hedging patterns
PLUS_MIXED = "#E76F51"    # burnt coral — dominant failure
PLUS_MINUS = "#F4A261"    # sandy orange
PLUS_NONE = "#E9C46A"     # gold
OTHER = "#B0BEC5"         # cool gray (all other patterns aggregated)

tasks = ["Task1\nEcon", "Task1\nFinance", "Task2", "Task3"]

# From Appendix E table: top regression patterns
plus_mixed = [96, 95, 9, 34]
plus_minus = [37, 41, 5, 36]
plus_none = [25, 20, 0, 0]   # Task2/Task3 not reported, treated as 0

# Other regressions = total regressions minus the three patterns above
# Task1 Econ: 182 total regressions; Task1 Fin: 174; Task2: ~39; Task3: ~92
total_regressions = [182, 174, 39, 92]
other = [t - sum(p) for t, p in zip(total_regressions,
         [[a+b+c] for a,b,c in zip(plus_mixed, plus_minus, plus_none)])]
other = [max(0, v) for v in other]  # safety clamp

fig, ax = plt.subplots(figsize=(5.5, 3.5))

x = np.arange(len(tasks))
width = 0.55

# Stacked bars
b1 = ax.bar(x, plus_mixed, width, label="+ → mixed (hedging)",
            color=PLUS_MIXED, edgecolor="white", linewidth=0.5, zorder=3)
b2 = ax.bar(x, plus_minus, width, bottom=plus_mixed,
            label="+ → − (flipping)", color=PLUS_MINUS,
            edgecolor="white", linewidth=0.5, zorder=3)
b3 = ax.bar(x, plus_none, width,
            bottom=[a+b for a,b in zip(plus_mixed, plus_minus)],
            label="+ → none (nullifying)", color=PLUS_NONE,
            edgecolor="white", linewidth=0.5, zorder=3)
b4 = ax.bar(x, other, width,
            bottom=[a+b+c for a,b,c in zip(plus_mixed, plus_minus, plus_none)],
            label="Other patterns", color=OTHER,
            edgecolor="white", linewidth=0.5, zorder=3)

# Percentage annotations on +→mixed (dominant segment)
for bar, val, total in zip(b1, plus_mixed, total_regressions):
    pct = val / total * 100
    mid = bar.get_y() + bar.get_height() / 2
    ax.annotate(f"{pct:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, mid),
                ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")

# Total count on top
for xi, tot in zip(x, total_regressions):
    ax.annotate(f"n={tot}",
                xy=(xi, tot + 2), ha="center", va="bottom",
                fontsize=7, color="#555", style="italic")

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=9)
ax.set_ylabel("Number of regression samples")
ax.set_ylim(0, max(total_regressions) * 1.15)
ax.legend(ncol=2, loc="upper right", fontsize=7)
ax.set_title("EconCausal: dominant regression patterns", pad=8)

fig.savefig("paper/iclr2026/figures/fig_hedging_patterns.pdf")
fig.savefig("paper/iclr2026/figures/fig_hedging_patterns.png", dpi=300)
print("Saved: fig_hedging_patterns.pdf, fig_hedging_patterns.png")
