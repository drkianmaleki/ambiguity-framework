"""
experiments/run_aamass_tstar.py — AAMass at F1-optimal threshold t*.

Addresses Reviewer 1 Major Comment #1:
    "On heavily imbalanced datasets, evaluating AAMass at 0.5 yields
    near-zero values — an artifact of poor threshold choice, not
    genuine task certainty. Demonstrate AAMass at an operationally
    realistic threshold t* optimised for F1."

For each dataset and classifier, this script:
    1. Finds t* = argmax_t F1(t) over a fine threshold grid
    2. Evaluates AAMass at [t* - δ, t* + δ]  (threshold-centred interval)
    3. Compares against AAMass at the default [0.5 - δ, 0.5 + δ]
    4. Reports how much information is recovered by using t*

Datasets evaluated:
    All six, with emphasis on imbalanced cases:
    - creditcard  (p = 0.0017, extreme)
    - nslkdd      (p = 0.800,  artificially balanced but threshold far from 0.5)
    - brfss       (p = 0.094,  moderate imbalance)

Output
------
results/tables/aamass_tstar.csv          — full results
results/tables/aamass_tstar_summary.csv  — comparison table for paper

Usage
-----
    python experiments/run_aamass_tstar.py
    python experiments/run_aamass_tstar.py --datasets creditcard nslkdd brfss
    python experiments/run_aamass_tstar.py --delta 0.05 0.10 0.15
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import load_dataset

RESULTS_DIR  = PROJECT_ROOT / "results" / "tables"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
GRID_STEPS   = 200      # threshold grid resolution
DATASETS     = ["heart", "uci_credit", "readmissions",
                "nslkdd", "brfss", "creditcard"]
DELTAS       = [0.05, 0.10, 0.15, 0.20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tstar(y_true: np.ndarray, y_prob: np.ndarray,
               grid_steps: int = GRID_STEPS) -> tuple:
    """
    Find t* = argmax_t F1(t) over a uniform threshold grid.

    Returns
    -------
    t_star   : float — F1-optimal threshold
    f1_star  : float — F1 score at t*
    """
    thresholds = np.linspace(0.01, 0.99, grid_steps)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 4), round(float(best_f1), 4)


def aamass_at_threshold(y_prob: np.ndarray,
                        center: float,
                        delta: float) -> float:
    """
    Fraction of predictions in [center - delta, center + delta].
    """
    lo, hi = center - delta, center + delta
    return round(float(((y_prob >= lo) & (y_prob <= hi)).mean()), 4)


def build_classifiers() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        ),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_dataset(name: str, X: np.ndarray, y: np.ndarray,
                deltas: list) -> pd.DataFrame:

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    y_te   = np.array(y_te)

    prevalence = float(y_te.mean())
    rows = []

    print(f"\n  Dataset: {name}  "
          f"(N={len(y):,}  prevalence={prevalence:.4f})",
          flush=True)

    for clf_name, clf in build_classifiers().items():
        import copy
        clf_ = copy.deepcopy(clf)
        t0 = time.perf_counter()
        clf_.fit(X_tr, y_tr)
        y_prob = clf_.predict_proba(X_te)[:, 1]
        elapsed = time.perf_counter() - t0

        auc    = round(roc_auc_score(y_te, y_prob), 4)
        t_star, f1_star = find_tstar(y_te, y_prob)

        for delta in deltas:
            aa_default = aamass_at_threshold(y_prob, 0.5,    delta)
            aa_tstar   = aamass_at_threshold(y_prob, t_star, delta)
            gain       = round(aa_tstar - aa_default, 4)

            rows.append({
                "dataset":        name,
                "classifier":     clf_name,
                "prevalence":     round(prevalence, 4),
                "delta":          delta,
                "t_star":         t_star,
                "f1_at_tstar":    f1_star,
                "aamass_default": aa_default,   # centred at 0.5
                "aamass_tstar":   aa_tstar,     # centred at t*
                "gain":           gain,          # tstar - default
                "auc_roc":        auc,
                "fit_time_s":     round(elapsed, 3),
            })

        print(
            f"    {clf_name:20s}  t*={t_star:.3f}  "
            f"F1@t*={f1_star:.3f}  "
            f"AAMass(0.5)={aamass_at_threshold(y_prob, 0.5, 0.10):.4f}  "
            f"AAMass(t*)={aamass_at_threshold(y_prob, t_star, 0.10):.4f}  "
            f"AUC={auc:.4f}",
            flush=True,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AAMass at F1-optimal threshold t*."
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS,
    )
    parser.add_argument(
        "--delta", type=float, nargs="+", default=DELTAS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  AAMass at F1-Optimal Threshold t*")
    print(f"  Datasets : {args.datasets}")
    print(f"  Deltas   : {args.delta}")
    print("=" * 70)

    all_rows = []
    for name in args.datasets:
        try:
            X, y, _ = load_dataset(name)
            df = run_dataset(name, X, y, args.delta)
            all_rows.append(df)
        except Exception as e:
            print(f"  ERROR on {name}: {e}")

    if not all_rows:
        print("No results.")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # Full results
    out = RESULTS_DIR / "aamass_tstar.csv"
    combined.to_csv(out, index=False)

    # Summary at delta=0.10
    summary = (
        combined[combined["delta"] == 0.10]
        .groupby(["dataset", "classifier"])[
            ["prevalence", "t_star", "f1_at_tstar",
             "aamass_default", "aamass_tstar", "gain", "auc_roc"]
        ]
        .first()
        .round(4)
    )
    sum_out = RESULTS_DIR / "aamass_tstar_summary.csv"
    summary.to_csv(sum_out)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Full results → {out}")
    print(f"  Summary      → {sum_out}")
    print()
    print("Summary at δ = 0.10:")
    print(summary.to_string())
    print()
    print("Key gain column: aamass_tstar − aamass_default")
    print("Positive gain = t* recovers information lost at default 0.5")
    print("=" * 70)


if __name__ == "__main__":
    main()
