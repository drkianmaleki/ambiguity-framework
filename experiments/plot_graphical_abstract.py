"""
experiments/plot_graphical_abstract.py — Graphical abstract (v4).

Layout: title bar + 4 content columns. No caption row. No bar chart.
All text is contained within its own GridSpec cell — no bleed.

+──────────────────── TITLE (full width, navy) ────────────────────+
│  Paper title · Author                                             │
+──────────┬──────────────┬────────────────┬──────────────────────+
│ PROBLEM  │  FRAMEWORK   │  KEY FINDING   │      TAKEAWAYS       │
│ hist A   │  AA_Mass box │  hero +58%     │  6 datasets          │
│ hist B   │  AA_PR  box  │  0.376→0.595   │  r=0.72 entropy      │
│          │  AA_F1  box  │  AUC unchanged │  non-overlap. CIs    │
│          │              │  after calib.  │  Theorem 2 proven    │
+──────────┴──────────────┴────────────────┴──────────────────────+

Output:  results/figures/graphical_abstract.pdf  /  .png
Usage:   python experiments/plot_graphical_abstract.py
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR  = PROJECT_ROOT / "results" / "figures"

DPI = 150
W   = 1328 / DPI
H   = 560  / DPI

# Palette
BG    = "#FFFFFF"
NAVY  = "#0d2137"
BLUE  = "#2166ac"
RED   = "#d6604d"
GREEN = "#4dac26"
AMBER = "#e8a020"
LGRAY = "#f0f4f8"
MGRAY = "#c8d4e0"
DGRAY = "#6b7280"
WHITE = "#FFFFFF"


def callout_cell(ax, number, label, color):
    """Draw a number callout inside an existing axes."""
    ax.set_facecolor(color + "15")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.axvline(0.03, color=color, lw=5, solid_capstyle="butt")
    ax.text(0.52, 0.64, number,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=color)
    ax.text(0.52, 0.22, label,
            ha="center", va="center",
            fontsize=7.0, color=NAVY, linespacing=1.4)


def metric_cell(ax, symbol, description, tag, color):
    """Draw a metric description box inside an existing axes."""
    ax.set_facecolor(color + "15")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.axvline(0.03, color=color, lw=5, solid_capstyle="butt")
    ax.text(0.09, 0.72, symbol,
            ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=color)
    ax.text(0.09, 0.40, description,
            ha="left", va="center",
            fontsize=7.0, color=NAVY, linespacing=1.5)
    ax.text(0.09, 0.12, tag,
            ha="left", va="center",
            fontsize=6.5, color=DGRAY, style="italic")


def section_header(ax, label, color):
    """Draw a coloured section header inside an existing axes."""
    ax.set_facecolor(color + "25")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0.5, 0.50, label,
            ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=color)


def make_figure():
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)

    # ── Outer grid: 2 rows × 4 cols ───────────────────────────────────────
    # Row 0: title bar
    # Row 1: four content columns
    outer = gridspec.GridSpec(
        2, 4,
        figure=fig,
        left=0.01, right=0.99,
        top=0.99,  bottom=0.01,
        hspace=0.03, wspace=0.02,
        height_ratios=[0.20, 0.80],
        width_ratios=[0.22, 0.26, 0.24, 0.28],
    )

    # ── Title bar (merged all 4 cols) ─────────────────────────────────────
    ax_title = fig.add_subplot(outer[0, :])
    ax_title.set_facecolor(WHITE)
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
    ax_title.set_axis_off()
    ax_title.axhline(0.0, color=NAVY, lw=2)
    ax_title.text(0.5, 0.75,
                  "The Ambiguity Range Framework",
                  ha="center", va="center",
                  fontsize=20, fontweight="bold", color="black")
    ax_title.text(0.5, 0.40,
                  "A Diagnostic Toolkit for Operational Evaluation of Binary Classifiers",
                  ha="center", va="center",
                  fontsize=9, color=DGRAY)
    ax_title.text(0.5, 0.12,
                  "K. Maleki  ·  Creighton University",
                  ha="center", va="center",
                  fontsize=8, color=DGRAY)

    # ═══════════════════════════════════════════════════════════════════════
    # COL 0 — THE PROBLEM
    # Nested grid: header + hist A + hist B
    # ═══════════════════════════════════════════════════════════════════════
    col0 = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=outer[1, 0],
        hspace=0.06,
        height_ratios=[0.09, 0.455, 0.455],
    )

    section_header(fig.add_subplot(col0[0]), "THE PROBLEM", RED)

    np.random.seed(42)
    bins = np.linspace(0, 1, 17)

    for row_idx, (data, label, col, aamass_str, auc_str) in enumerate([
        (np.clip(np.random.beta(4, 4, 600), 0.01, 0.99),
         "Classifier A", BLUE, "AA_Mass = 0.53", "AUC = 0.92"),
        (np.clip(np.concatenate([np.random.beta(9, 2, 300),
                                  np.random.beta(2, 9, 300)]), 0.01, 0.99),
         "Classifier B", RED, "AA_Mass = 0.02", "AUC = 0.92"),
    ]):
        ax_h = fig.add_subplot(col0[row_idx + 1])
        ax_h.set_facecolor(LGRAY)
        ax_h.hist(data, bins=bins, density=True,
                  color=col, alpha=0.80, zorder=3)
        ax_h.axvspan(0.40, 0.60, alpha=0.28, color=AMBER, zorder=2)
        ax_h.axvline(0.5, color=AMBER, lw=1.2, ls="--", zorder=4)
        ax_h.set_xlim(0, 1)
        ax_h.set_yticks([])
        ax_h.set_xticks([0, 0.5, 1])
        ax_h.tick_params(labelsize=7, colors="black")
        ax_h.spines[["top", "right", "left"]].set_visible(False)
        ax_h.spines["bottom"].set_edgecolor(MGRAY)
        ax_h.set_ylabel("Density", fontsize=7.5, color="black")

        if row_idx == 0:
            ax_h.set_xticklabels([])
        else:
            ax_h.set_xlabel("Predicted probability", fontsize=7.5, color="black")

        ax_h.text(0.03, 0.93, label, transform=ax_h.transAxes,
                  ha="left", va="top", fontsize=6.5,
                  fontweight="bold", color="black")
        ax_h.text(0.97, 0.93, aamass_str, transform=ax_h.transAxes,
                  ha="right", va="top", fontsize=5.8,
                  fontweight="bold", color="black")
        ax_h.text(0.97, 0.65, auc_str, transform=ax_h.transAxes,
                  ha="right", va="top", fontsize=5.5, color="black")

    # ═══════════════════════════════════════════════════════════════════════
    # COL 1 — THE FRAMEWORK
    # Nested grid: header + 3 metric cells
    # ═══════════════════════════════════════════════════════════════════════
    col1 = gridspec.GridSpecFromSubplotSpec(
        4, 1,
        subplot_spec=outer[1, 1],
        hspace=0.05,
        height_ratios=[0.09, 0.30, 0.30, 0.31],
    )

    section_header(fig.add_subplot(col1[0]), "THE FRAMEWORK", BLUE)

    metrics = [
        (r"$\mathrm{AA_{Mass}}(\delta)$",
         "Fraction of predictions\ninside ℐ_δ = [0.5−δ,  0.5+δ]",
         "Local  ·  threshold-sensitive", BLUE),
        (r"$\mathrm{AA_{PR}}$",
         "Precision·recall degradation\nacross all thresholds",
         "Global  ·  delta-invariant", RED),
        (r"$\mathrm{AA_{F1}}$",
         "F_β degradation across\nall thresholds",
         "Global  ·  extends to F_β", GREEN),
    ]
    for i, (sym, desc, tag, col) in enumerate(metrics):
        metric_cell(fig.add_subplot(col1[i + 1]), sym, desc, tag, col)

    # ═══════════════════════════════════════════════════════════════════════
    # COL 2 — KEY FINDING  (pure callout boxes — no axes)
    # Nested grid: header + 4 callout cells
    # ═══════════════════════════════════════════════════════════════════════
    col2 = gridspec.GridSpecFromSubplotSpec(
        5, 1,
        subplot_spec=outer[1, 2],
        hspace=0.05,
        height_ratios=[0.09, 0.23, 0.23, 0.23, 0.22],
    )

    section_header(fig.add_subplot(col2[0]), "KEY FINDING", GREEN)

    key_callouts = [
        ("+58%",       "AA_Mass after\nPlatt calibration",   GREEN),
        ("0.376→0.595","XGBoost raw\nvs calibrated",         BLUE),
        ("AUC flat",   "0.632→0.641\nno discrimination gain", DGRAY),
        ("Theorem 2",  "calibration linkage\nformally proven", NAVY),
    ]
    for i, (num, lbl, col) in enumerate(key_callouts):
        callout_cell(fig.add_subplot(col2[i + 1]), num, lbl, col)

    # ═══════════════════════════════════════════════════════════════════════
    # COL 3 — TAKEAWAYS  (pure callout boxes — no axes)
    # Nested grid: header + 4 callout cells
    # ═══════════════════════════════════════════════════════════════════════
    col3 = gridspec.GridSpecFromSubplotSpec(
        5, 1,
        subplot_spec=outer[1, 3],
        hspace=0.05,
        height_ratios=[0.09, 0.23, 0.23, 0.23, 0.22],
    )

    section_header(fig.add_subplot(col3[0]), "TAKEAWAYS", AMBER)

    takeaways = [
        ("6",        "real-world\ndatasets validated",         BLUE),
        ("r = 0.72", "AA_Mass vs entropy\nnon-redundant",      RED),
        ("p < 0.05", "non-overlapping\nbootstrap CIs",         GREEN),
        ("3",        "formal theorems\nwith full proofs",      AMBER),
    ]
    for i, (num, lbl, col) in enumerate(takeaways):
        callout_cell(fig.add_subplot(col3[i + 1]), num, lbl, col)

    return fig


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating graphical abstract (v4)...")
    fig = make_figure()
    pdf = FIGURES_DIR / "graphical_abstract.pdf"
    png = FIGURES_DIR / "graphical_abstract.png"
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight", facecolor=BG)
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"PDF → {pdf}")
    print(f"PNG → {png}")
    try:
        from PIL import Image
        img = Image.open(png)
        print(f"Size: {img.size[0]}w × {img.size[1]}h px  (required ≥1328 × ≥531)")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
