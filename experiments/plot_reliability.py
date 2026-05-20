"""
experiments/plot_reliability.py — Generate reliability diagrams for the paper.

Produces a publication-quality reliability diagram (calibration plot) for a
specified dataset, showing calibration curves for all three classifiers
alongside the perfect calibration diagonal.

The figure is saved to results/figures/reliability_<dataset>.pdf (for LaTeX)
and results/figures/reliability_<dataset>.png (for preview).

Usage
-----
    python experiments/plot_reliability.py                        # readmissions
    python experiments/plot_reliability.py --dataset heart
    python experiments/plot_reliability.py --dataset uci_credit
    python experiments/plot_reliability.py --all                  # all 6 datasets
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import load_dataset

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
N_BINS      = 10
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Publication-quality style settings
CLASSIFIERS = {
    "Logistic Regression": {
        "color": "#2166ac",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 7,
    },
    "Random Forest": {
        "color": "#d6604d",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.0,
        "markersize": 7,
    },
    "XGBoost": {
        "color": "#4dac26",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.0,
        "markersize": 7,
    },
}


def build_classifiers() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = N_BINS) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 \
               else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(float(y_true[mask].mean()) -
                                       float(y_prob[mask].mean()))
    return float(ece)


def plot_reliability(dataset_name: str) -> Path:
    """
    Fit all three classifiers on the dataset and produce a reliability diagram.

    Returns the path to the saved PDF.
    """
    print(f"  Loading {dataset_name}...")
    X, y, _ = load_dataset(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1,
        figsize=(6, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # --- Perfect calibration diagonal ---
    ax_cal.plot(
        [0, 1], [0, 1],
        linestyle=":",
        color="0.5",
        linewidth=1.5,
        label="Perfect calibration",
        zorder=1,
    )

    classifiers = build_classifiers()
    all_probs   = {}

    for clf_name, clf in classifiers.items():
        print(f"    Fitting {clf_name}...", end=" ", flush=True)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        all_probs[clf_name] = y_prob

        prob_true, prob_pred = calibration_curve(
            y_test, y_prob, n_bins=N_BINS, strategy="uniform"
        )

        brier = brier_score_loss(y_test, y_prob)
        ece   = compute_ece(y_test, y_prob)

        style = CLASSIFIERS[clf_name]
        label = (
            f"{clf_name}  "
            f"(Brier={brier:.3f}, ECE={ece:.3f})"
        )

        ax_cal.plot(
            prob_pred, prob_true,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            label=label,
            zorder=3,
        )
        print("done")

    # --- Calibration plot formatting ---
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.set_ylabel("Fraction of positives", fontsize=11)
    ax_cal.set_xticklabels([])
    ax_cal.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax_cal.set_title(
        f"Reliability Diagram — {dataset_name.replace('_', ' ').title()}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax_cal.grid(True, alpha=0.3, linewidth=0.6)
    ax_cal.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_cal.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # --- Histogram of predicted probabilities ---
    bins = np.linspace(0, 1, N_BINS + 1)
    for clf_name, y_prob in all_probs.items():
        style = CLASSIFIERS[clf_name]
        ax_hist.hist(
            y_prob,
            bins=bins,
            histtype="step",
            color=style["color"],
            linewidth=1.5,
            density=True,
        )

    ax_hist.set_xlim(-0.02, 1.02)
    ax_hist.set_xlabel("Mean predicted probability", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=9)
    ax_hist.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_hist.grid(True, alpha=0.3, linewidth=0.6)
    ax_hist.set_yscale("log")

    # --- Save ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"reliability_{dataset_name}.pdf"
    png_path = FIGURES_DIR / f"reliability_{dataset_name}.png"

    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"  Saved: {pdf_path}")
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reliability diagrams for the paper."
    )
    parser.add_argument(
        "--dataset", type=str, default="readmissions",
        help="Dataset name (default: readmissions)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate diagrams for all 6 datasets"
    )
    args = parser.parse_args()

    datasets = (
        ["heart", "uci_credit", "readmissions",
         "nslkdd", "brfss", "creditcard"]
        if args.all else [args.dataset]
    )

    print(f"\nGenerating reliability diagram(s): {datasets}\n")
    for name in datasets:
        plot_reliability(name)

    print("\nDone. Files saved to results/figures/")
    print("\nTo include in LaTeX:")
    print("  \\begin{figure}[ht]")
    print("  \\centering")
    print(f"  \\includegraphics[width=0.75\\linewidth]{{figures/reliability_{datasets[0]}.pdf}}")
    print(f"  \\caption{{Reliability diagram for {datasets[0].replace('_',' ').title()}. ...}}")
    print("  \\end{figure}")


if __name__ == "__main__":
    main()
