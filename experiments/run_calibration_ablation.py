"""
experiments/run_calibration_ablation.py — Calibration ablation study.

For each dataset and classifier, compares ambiguity metrics and calibration
scores under three probability estimation strategies:

  raw         — uncalibrated probabilities (default classifier output)
  platt       — Platt scaling (sigmoid calibration via CalibratedClassifierCV)
  isotonic    — isotonic regression calibration via CalibratedClassifierCV

The key prediction: post-calibration AUC should remain flat while
AAMass increases (because calibration spreads probability mass toward 0.5
on genuinely uncertain instances). This directly validates Theorem 2
(calibration linkage).

Output
------
results/tables/calibration_ablation.csv
results/tables/calibration_ablation_summary.csv

Usage
-----
    python experiments/run_calibration_ablation.py
    python experiments/run_calibration_ablation.py --datasets heart readmissions
    python experiments/run_calibration_ablation.py --delta 0.10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import load_dataset
from ambiguity_suite import compute_AAPR, compute_AAF1, compute_AAMass

RESULTS_DIR  = PROJECT_ROOT / "results" / "tables"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
ECE_N_BINS   = 10

DATASETS = ["heart", "uci_credit", "readmissions", "nslkdd", "brfss", "creditcard"]

CALIBRATION_METHODS = ["raw", "platt", "isotonic"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = ECE_N_BINS) -> float:
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
    return round(float(ece), 4)


def build_base_classifiers() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0
        ),
    }


def get_probabilities(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method: str,
) -> np.ndarray:
    """
    Fit classifier and return predicted probabilities for the test set.

    For 'platt' and 'isotonic', wraps the base classifier in
    CalibratedClassifierCV using 5-fold cross-validation on the training set.

    Parameters
    ----------
    clf    : base sklearn-compatible classifier (unfitted)
    method : 'raw', 'platt', or 'isotonic'

    Returns
    -------
    y_prob : (n_test,) float array
    """
    if method == "raw":
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]

    # Platt or isotonic — use CalibratedClassifierCV
    cal_method = "sigmoid" if method == "platt" else "isotonic"
    calibrated = CalibratedClassifierCV(clf, method=cal_method, cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------------------------
# Core ablation loop
# ---------------------------------------------------------------------------

def run_ablation(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 0.10,
    beta: float = 1.0,
) -> pd.DataFrame:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    rows = []
    base_clfs = build_base_classifiers()
    total = len(base_clfs) * len(CALIBRATION_METHODS)
    done  = 0

    print(f"\n  Dataset: {dataset_name}  "
          f"(N={len(y):,}, prevalence={y.mean():.3f})")

    for clf_name, base_clf in base_clfs.items():
        for method in CALIBRATION_METHODS:
            t0 = time.perf_counter()

            # Need a fresh unfitted clone for each method
            import copy
            clf_copy = copy.deepcopy(base_clf)

            y_prob = get_probabilities(
                clf_copy, X_train, y_train, X_test, method
            )
            elapsed = round(time.perf_counter() - t0, 3)

            auc    = round(roc_auc_score(y_test, y_prob), 4)
            brier  = round(brier_score_loss(y_test, y_prob), 4)
            ece    = compute_ece(y_test, y_prob)
            aapr   = round(compute_AAPR(y_test, y_prob), 4)
            aaf1   = round(compute_AAF1(y_test, y_prob, beta=beta), 4)
            aamass = round(compute_AAMass(y_test, y_prob, delta=delta), 4)

            rows.append({
                "dataset":     dataset_name,
                "classifier":  clf_name,
                "calibration": method,
                "delta":       delta,
                "aapr":        aapr,
                "aaf1":        aaf1,
                "aamass":      aamass,
                "auc_roc":     auc,
                "brier":       brier,
                "ece":         ece,
                "fit_time_s":  elapsed,
            })

            done += 1
            pct = done / total * 100
            print(
                f"    [{pct:5.1f}%] {clf_name:20s} | {method:9s} "
                f"→ AAMass={aamass:.4f}  AUC={auc:.4f}  "
                f"Brier={brier:.4f}  ECE={ece:.4f}",
                flush=True,
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibration ablation study for the Ambiguity Range Framework."
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS,
        help="Datasets to run (default: all six)"
    )
    parser.add_argument(
        "--delta", type=float, default=0.10,
        help="Delta value for AAMass (default: 0.10)"
    )
    parser.add_argument(
        "--force-reload", action="store_true",
        help="Bypass dataset cache"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  Calibration Ablation Study")
    print(f"  Datasets : {args.datasets}")
    print(f"  Methods  : raw | platt | isotonic")
    print(f"  Delta    : {args.delta}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    failed  = []

    wall_start = time.time()

    for name in args.datasets:
        try:
            X, y, _ = load_dataset(name, force_reload=args.force_reload)
            df = run_ablation(name, X, y, delta=args.delta)
            all_dfs.append(df)
        except Exception as e:
            print(f"\n  ERROR on {name}: {e}")
            failed.append((name, str(e)))

    if not all_dfs:
        print("No results generated.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # --- Save full results ---
    full_path = RESULTS_DIR / "calibration_ablation.csv"
    combined.to_csv(full_path, index=False)

    # --- Save summary (mean across datasets per classifier × calibration) ---
    summary = (
        combined.groupby(["dataset", "classifier", "calibration"])[
            ["aapr", "aaf1", "aamass", "auc_roc", "brier", "ece"]
        ]
        .mean()
        .round(4)
    )

    summary_path = RESULTS_DIR / "calibration_ablation_summary.csv"
    summary.to_csv(summary_path)

    total_time = time.time() - wall_start
    print()
    print("=" * 70)
    print(f"  Done in {total_time/60:.1f} min")
    print(f"  Full results  → {full_path}")
    print(f"  Summary       → {summary_path}")
    print()
    print("Summary (mean across datasets):")
    print()

    # Print a clean pivot: classifier × calibration method
    for clf in combined["classifier"].unique():
        sub = combined[combined["classifier"] == clf]
        pivot = sub.groupby("calibration")[
            ["aamass", "auc_roc", "brier", "ece"]
        ].mean().round(4)
        print(f"  {clf}")
        print(pivot.to_string())
        print()

    print("=" * 70)

    if failed:
        print("\nFailed datasets:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
