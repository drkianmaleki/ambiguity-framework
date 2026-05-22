"""
experiments/run_entropy_comparison.py — Predictive entropy and margin
uncertainty baseline comparison.

Computes three uncertainty measures alongside AAMass for all classifiers
and datasets, then demonstrates the key case where AAMass diverges from
entropy — directly answering "why not just use predictive entropy?"

Measures
--------
  entropy   : mean predictive entropy H(p) = -p log p - (1-p) log(1-p)
  margin    : mean margin from decision boundary 1 - |2p - 1|
              (= 1 when p=0.5, = 0 when p=0 or p=1)
  aamass    : fraction of predictions in [0.5-δ, 0.5+δ]

Key distinction:
  A bimodal classifier with predictions at 0.1 and 0.9 has HIGH entropy
  but near-ZERO AAMass. Entropy cannot distinguish this from a classifier
  with all predictions at 0.5 — but the operational implications differ
  completely.

Output
------
results/tables/entropy_comparison.csv
results/tables/entropy_comparison_summary.csv

Usage
-----
    python experiments/run_entropy_comparison.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import load_dataset
from ambiguity_suite import compute_AAMass, compute_AAPR

RESULTS_DIR  = PROJECT_ROOT / "results" / "tables"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
DELTA        = 0.10
DATASETS     = ["heart", "uci_credit", "readmissions",
                "nslkdd", "brfss", "creditcard"]


# ---------------------------------------------------------------------------
# Uncertainty measures
# ---------------------------------------------------------------------------

def predictive_entropy(y_prob: np.ndarray) -> float:
    """Mean binary predictive entropy H(p) = -p log p - (1-p) log(1-p)."""
    eps = 1e-10
    p = np.clip(y_prob, eps, 1 - eps)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return round(float(h.mean()), 4)


def mean_margin(y_prob: np.ndarray) -> float:
    """
    Mean margin uncertainty: 1 - |2p - 1|.
    = 1 when p = 0.5 (maximum uncertainty at boundary)
    = 0 when p = 0 or 1 (maximum confidence)
    """
    return round(float((1 - np.abs(2 * y_prob - 1)).mean()), 4)


def fraction_near_boundary(y_prob: np.ndarray, delta: float = DELTA) -> float:
    """AAMass: fraction of predictions in [0.5-delta, 0.5+delta]."""
    return round(float(((y_prob >= 0.5 - delta) & (y_prob <= 0.5 + delta)).mean()), 4)


def build_classifiers() -> dict:
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


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison() -> pd.DataFrame:
    rows = []

    for dataset_name in DATASETS:
        print(f"  {dataset_name}...", flush=True)
        try:
            X, y, _ = load_dataset(dataset_name)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y
        )
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)
        y_te   = np.array(y_te)

        for clf_name, clf in build_classifiers().items():
            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_te)[:, 1]

            ent    = predictive_entropy(y_prob)
            margin = mean_margin(y_prob)
            aamass = fraction_near_boundary(y_prob)
            auc    = round(roc_auc_score(y_te, y_prob), 4)
            aapr   = round(compute_AAPR(y_te, y_prob), 4)

            rows.append({
                "dataset":    dataset_name,
                "classifier": clf_name,
                "entropy":    ent,
                "margin":     margin,
                "aamass":     aamass,
                "aapr":       aapr,
                "auc_roc":    auc,
            })

            print(
                f"    {clf_name:20s} "
                f"Entropy={ent:.4f}  Margin={margin:.4f}  "
                f"AAMass={aamass:.4f}  AUC={auc:.4f}",
                flush=True,
            )

    return pd.DataFrame(rows)


def analyse_divergence(df: pd.DataFrame) -> None:
    """
    Identify cases where entropy and AAMass rankings disagree —
    this is the core argument for why AAMass is not redundant.
    """
    print()
    print("=" * 70)
    print("Divergence analysis: where entropy and AAMass disagree")
    print("=" * 70)

    for dataset_name in df["dataset"].unique():
        sub = df[df["dataset"] == dataset_name].copy()
        if len(sub) < 2:
            continue

        # Rank by entropy (high = more uncertain) and by AAMass (high = more ambiguous)
        sub["entropy_rank"] = sub["entropy"].rank(ascending=False)
        sub["aamass_rank"]  = sub["aamass"].rank(ascending=False)
        sub["rank_diff"]    = (sub["entropy_rank"] - sub["aamass_rank"]).abs()

        max_diff = sub["rank_diff"].max()
        if max_diff >= 1:
            print(f"\n  {dataset_name} (rank disagreement = {max_diff:.0f}):")
            for _, row in sub.iterrows():
                print(f"    {row['classifier']:20s}  "
                      f"Entropy={row['entropy']:.4f} (rank {row['entropy_rank']:.0f})  "
                      f"AAMass={row['aamass']:.4f} (rank {row['aamass_rank']:.0f})")

    print()
    print("Correlation between entropy and AAMass across all rows:")
    corr = df["entropy"].corr(df["aamass"])
    print(f"  Pearson r = {corr:.3f}")
    print()
    print("Interpretation:")
    print("  r < 0.7: AAMass captures information entropy does not")
    print("  Divergences above show cases where classifier ranking differs")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  Entropy and Margin Uncertainty Comparison")
    print(f"  Datasets : {DATASETS}")
    print(f"  Delta    : {DELTA}")
    print("=" * 70)
    print()

    df = run_comparison()

    # Save full results
    out = RESULTS_DIR / "entropy_comparison.csv"
    df.to_csv(out, index=False)

    # Save summary
    summary = (
        df.groupby(["dataset", "classifier"])[
            ["entropy", "margin", "aamass", "aapr", "auc_roc"]
        ]
        .mean()
        .round(4)
    )
    summary_out = RESULTS_DIR / "entropy_comparison_summary.csv"
    summary.to_csv(summary_out)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s → {out}")
    print()
    print("Summary:")
    print(summary.to_string())

    analyse_divergence(df)


if __name__ == "__main__":
    main()
