"""
experiments/run_paired_tests.py — Paired statistical tests across classifiers.

Runs 5-fold stratified cross-validation on all six datasets, computing
AAMass, AAPR, AAF1, and AUC for each fold. Then applies Wilcoxon signed-rank
tests on fold-paired observations to formally validate the claim:

    "Classifiers with similar AUC differ significantly in AAMass."

Why CV instead of a single test split:
    A single test set yields one scalar per classifier — unpaired and
    untestable. Five folds yield five paired observations (same data
    partitions for all classifiers), enabling non-parametric paired tests
    with no distributional assumptions.

Statistical tests:
    Wilcoxon signed-rank test (scipy.stats.wilcoxon)
    Effect size: rank-biserial correlation r = 1 - 2W / (n(n+1)/2)
    Significance threshold: α = 0.05 (Bonferroni-corrected per dataset)

Key comparisons reported:
    1. Heart: RF vs XGBoost AAMass (AUC tied at ~0.923)
    2. Readmissions: LR vs XGBoost AAMass
    3. All pairwise classifier comparisons per dataset

Output
------
results/tables/paired_tests_cv.csv       — fold-level CV results
results/tables/paired_tests_stats.csv    — Wilcoxon test results
results/tables/paired_tests_summary.csv  — key comparisons for paper

Usage
-----
    python experiments/run_paired_tests.py
    python experiments/run_paired_tests.py --datasets heart readmissions
    python experiments/run_paired_tests.py --folds 5
"""

import argparse
import sys
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import load_dataset
from ambiguity_suite import compute_AAMass, compute_AAPR, compute_AAF1

RESULTS_DIR  = PROJECT_ROOT / "results" / "tables"
RANDOM_STATE = 42
N_FOLDS      = 10
DELTA        = 0.10
ALPHA        = 0.05
DATASETS     = ["heart", "uci_credit", "readmissions",
                "nslkdd", "brfss", "creditcard"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            random_state=RANDOM_STATE, verbosity=0,
            n_jobs=-1,
        ),
    }


def rank_biserial(w_stat: float, n: int) -> float:
    """Effect size for Wilcoxon signed-rank test."""
    max_w = n * (n + 1) / 2
    return float(1 - 2 * w_stat / max_w)


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank test on paired arrays a and b.

    Returns dict with: statistic, p_value, effect_size, n, significant
    """
    diff = a - b
    # Remove zero differences
    diff = diff[diff != 0]
    n = len(diff)

    if n < 4:
        return {
            "statistic": np.nan, "p_value": np.nan,
            "effect_size": np.nan, "n": n, "significant": False,
            "note": "too few non-zero differences"
        }

    try:
        result = stats.wilcoxon(a, b, alternative="two-sided")
        r = rank_biserial(result.statistic, n)
        return {
            "statistic":   round(float(result.statistic), 4),
            "p_value":     round(float(result.pvalue), 4),
            "effect_size": round(r, 4),
            "n":           n,
            "significant": bool(result.pvalue < ALPHA),
            "note":        "",
        }
    except Exception as e:
        return {
            "statistic": np.nan, "p_value": np.nan,
            "effect_size": np.nan, "n": n, "significant": False,
            "note": str(e)
        }


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def run_cv(dataset_name: str, X: np.ndarray, y: np.ndarray,
           n_folds: int) -> pd.DataFrame:
    """
    Run stratified k-fold CV for all classifiers on one dataset.
    Returns a DataFrame with one row per (fold, classifier).
    """
    skf  = StratifiedKFold(n_splits=n_folds, shuffle=True,
                           random_state=RANDOM_STATE)
    rows = []
    classifiers = build_classifiers()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        print(f"    Fold {fold_idx + 1}/{n_folds}  "
              f"(train={len(y_tr):,}  test={len(y_te):,})",
              flush=True)

        for clf_name, clf in classifiers.items():
            import copy
            t0    = time.perf_counter()
            clf_  = copy.deepcopy(clf)
            clf_.fit(X_tr, y_tr)
            y_prob = clf_.predict_proba(X_te)[:, 1]
            elapsed = time.perf_counter() - t0

            aamass = round(compute_AAMass(y_te, y_prob, delta=DELTA), 4)
            aapr   = round(compute_AAPR(y_te, y_prob), 4)
            aaf1   = round(compute_AAF1(y_te, y_prob), 4)
            auc    = round(roc_auc_score(y_te, y_prob), 4)

            rows.append({
                "dataset":    dataset_name,
                "fold":       fold_idx + 1,
                "classifier": clf_name,
                "aamass":     aamass,
                "aapr":       aapr,
                "aaf1":       aaf1,
                "auc_roc":    auc,
                "n_train":    len(y_tr),
                "n_test":     len(y_te),
                "fit_time_s": round(elapsed, 3),
            })

            print(
                f"      {clf_name:20s} "
                f"AAMass={aamass:.4f}  AUC={auc:.4f}  "
                f"AAPR={aapr:.4f}  ({elapsed:.1f}s)",
                flush=True,
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def run_tests(cv_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all pairwise Wilcoxon signed-rank tests per dataset and metric.

    Returns:
        stats_df   — full test results (all pairs, all metrics, all datasets)
        summary_df — key comparisons highlighted in the paper
    """
    clf_names = cv_df["classifier"].unique().tolist()
    pairs     = list(itertools.combinations(clf_names, 2))
    metrics   = ["aamass", "aapr", "aaf1", "auc_roc"]

    stat_rows    = []
    summary_rows = []

    # Bonferroni correction: n_comparisons per dataset
    n_comparisons = len(pairs) * len(metrics)
    alpha_corrected = ALPHA / n_comparisons

    for dataset_name in cv_df["dataset"].unique():
        ds = cv_df[cv_df["dataset"] == dataset_name]

        for clf_a, clf_b in pairs:
            for metric in metrics:
                a_vals = ds[ds["classifier"] == clf_a].sort_values("fold")[metric].values
                b_vals = ds[ds["classifier"] == clf_b].sort_values("fold")[metric].values

                result = wilcoxon_test(a_vals, b_vals)

                mean_a = round(float(a_vals.mean()), 4)
                mean_b = round(float(b_vals.mean()), 4)

                stat_rows.append({
                    "dataset":          dataset_name,
                    "clf_a":            clf_a,
                    "clf_b":            clf_b,
                    "metric":           metric,
                    "mean_a":           mean_a,
                    "mean_b":           mean_b,
                    "mean_diff":        round(mean_a - mean_b, 4),
                    "w_statistic":      result["statistic"],
                    "p_value":          result["p_value"],
                    "p_corrected":      round(result["p_value"] * n_comparisons, 4)
                                        if not np.isnan(result["p_value"]) else np.nan,
                    "effect_size_r":    result["effect_size"],
                    "significant":      result["significant"],
                    "sig_bonferroni":   (result["p_value"] < alpha_corrected)
                                        if not np.isnan(result["p_value"]) else False,
                    "n_folds":          result["n"],
                    "note":             result["note"],
                })

        # Key comparisons for paper
        key_pairs = [
            ("heart",        "RandomForest",       "XGBoost",             "aamass",
             "Heart: RF vs XGB AAMass (AUC tied)"),
            ("heart",        "RandomForest",       "XGBoost",             "auc_roc",
             "Heart: RF vs XGB AUC (should not differ)"),
            ("readmissions", "LogisticRegression", "XGBoost",             "aamass",
             "Readmissions: LR vs XGB AAMass"),
            ("readmissions", "LogisticRegression", "XGBoost",             "auc_roc",
             "Readmissions: LR vs XGB AUC"),
            ("uci_credit",   "LogisticRegression", "XGBoost",             "aamass",
             "UCI Credit: LR vs XGB AAMass"),
        ]

        for ds_name, clf_a, clf_b, metric, label in key_pairs:
            if ds_name != dataset_name:
                continue
            ds_sub = cv_df[cv_df["dataset"] == ds_name]
            a_vals = ds_sub[ds_sub["classifier"]==clf_a].sort_values("fold")[metric].values
            b_vals = ds_sub[ds_sub["classifier"]==clf_b].sort_values("fold")[metric].values
            result = wilcoxon_test(a_vals, b_vals)
            summary_rows.append({
                "comparison":    label,
                "metric":        metric,
                "mean_a":        round(float(a_vals.mean()), 4),
                "mean_b":        round(float(b_vals.mean()), 4),
                "mean_diff":     round(float(a_vals.mean()) - float(b_vals.mean()), 4),
                "p_value":       result["p_value"],
                "effect_size_r": result["effect_size"],
                "significant":   result["significant"],
            })

    return pd.DataFrame(stat_rows), pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon tests via stratified k-fold CV."
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS,
        help=f"Datasets to run (default: all {len(DATASETS)})"
    )
    parser.add_argument(
        "--folds", type=int, default=N_FOLDS,
        help=f"Number of CV folds (default: {N_FOLDS})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()

    print("=" * 70)
    print("  Paired Statistical Tests (Wilcoxon signed-rank)")
    print(f"  Datasets : {args.datasets}")
    print(f"  Folds    : {args.folds}-fold stratified CV")
    print(f"  Delta    : {DELTA}")
    print(f"  Alpha    : {ALPHA} (Bonferroni-corrected within dataset)")
    print("=" * 70)

    # ── Cross-validation ───────────────────────────────────────────────────
    all_cv = []
    for name in args.datasets:
        print(f"\n[CV] {name}...", flush=True)
        try:
            X, y, _ = load_dataset(name)
            df = run_cv(name, X, y, n_folds=args.folds)
            all_cv.append(df)
        except Exception as e:
            print(f"  ERROR: {e}")

    if not all_cv:
        print("No results. Exiting.")
        return

    cv_df = pd.concat(all_cv, ignore_index=True)
    cv_path = RESULTS_DIR / "paired_tests_cv.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"\nCV results saved → {cv_path}")

    # ── Statistical tests ──────────────────────────────────────────────────
    print("\nRunning Wilcoxon signed-rank tests...")
    stats_df, summary_df = run_tests(cv_df)

    stats_path   = RESULTS_DIR / "paired_tests_stats.csv"
    summary_path = RESULTS_DIR / "paired_tests_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    elapsed = (time.time() - t_wall) / 60
    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.1f} min")
    print(f"  CV data    → {cv_path}")
    print(f"  Full tests → {stats_path}")
    print(f"  Summary    → {summary_path}")
    print()
    print("Key comparisons for paper:")
    print("-" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
