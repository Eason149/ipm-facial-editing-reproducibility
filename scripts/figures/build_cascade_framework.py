#!/usr/bin/env python
"""Build the manuscript's non-causal cascade framework figure."""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("IPM_DATA_ROOT", ROOT.parent)).resolve()
LOCAL_PACKAGES = DATA_ROOT / "RSA_time_resolved_analysis" / ".python-packages"
if LOCAL_PACKAGES.exists():
    sys.path.append(str(LOCAL_PACKAGES))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, xy, width, height, text, color, fontsize=11, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.3, edgecolor="#333333", facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, linespacing=1.35)
    return patch


def arrow(ax, start, end, dashed=False, rad=0.0, label=None, label_xy=None):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.6,
        color="#303030", linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", shrinkA=5, shrinkB=5,
    )
    ax.add_patch(patch)
    if label and label_xy:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=9,
                color="#303030", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


fig, ax = plt.subplots(figsize=(11.5, 8.2), constrained_layout=True)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

box(ax, (0.25, 0.85), 0.50, 0.105,
    "Algorithmic visual transformation\n4 identities × FSlim / Eye / Mouth / Skin", "#eef4fb", 12, "bold")
box(ax, (0.055, 0.61), 0.39, 0.13,
    "Continuous magnitude\nG_new geometry / A2 local surface / I identity shift", "#e5f0fa", 10.5)
box(ax, (0.555, 0.61), 0.39, 0.13,
    "Semantic operation meaning\nFSlim / Eye / Mouth / Skin", "#f0e9fa", 10.5)
box(ax, (0.25, 0.36), 0.50, 0.12,
    "Temporally sensitive intermediate measure\nCentroparietal EEG: 350–600 / 600–1000 ms", "#fff0e5", 11)
box(ax, (0.30, 0.145), 0.40, 0.10,
    "Explicit evaluation\nBeauty / Naturalness", "#fff0e5", 11)
box(ax, (0.18, 0.018), 0.64, 0.065,
    "Cascade criterion: adjusted edge tests + held-out-participant prediction", "#eaf7e8", 10.5, "bold")

arrow(ax, (0.43, 0.85), (0.25, 0.74))
arrow(ax, (0.57, 0.85), (0.75, 0.74))
arrow(ax, (0.25, 0.61), (0.41, 0.48), label="Edge 1", label_xy=(0.31, 0.525))
arrow(ax, (0.75, 0.61), (0.59, 0.48), label="Edge 1", label_xy=(0.69, 0.525))
arrow(ax, (0.50, 0.36), (0.50, 0.245), label="Edge 2", label_xy=(0.56, 0.302))
arrow(ax, (0.055, 0.655), (0.30, 0.195), dashed=True, rad=-0.18,
      label="Direct evaluation information", label_xy=(0.13, 0.36))
arrow(ax, (0.945, 0.655), (0.70, 0.195), dashed=True, rad=0.18,
      label="Direct evaluation information", label_xy=(0.87, 0.36))
arrow(ax, (0.50, 0.145), (0.50, 0.083))

ax.text(0.5, 0.985, "Candidate representation–EEG–evaluation cascade",
        ha="center", va="top", fontsize=15, weight="bold")
ax.text(0.34, 0.105, "Association does not imply causal mediation",
        ha="center", va="center", fontsize=9.5, style="italic", color="#555555")

fig.savefig(OUT / "Figure_1_cascade_framework.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Figure_1_cascade_framework.pdf", bbox_inches="tight")
plt.close(fig)
