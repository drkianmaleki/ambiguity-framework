"""
experiments/run_bootstrap_ci.py — Bootstrap confidence intervals.

Computes 95% bootstrap confidence intervals for key comparisons:

  1. Synthetic: AUC-matched pair (variant A vs B) AAMass difference for LR
     Uses variance across the 5 seeds already in synthetic_results.csv.

  2. Real-world: Readmissions LR vs XGBoost AAMass difference
     Bootstraps (y_true, y_prob) test set pairs for N_BOOTSTRAP resamples.

  3. Real-world: All classifier × dataset AAMass and AAPR values
     Reports mean ± 95% CI for every cell in the main results table.

Output
------
results/tables/bootstrap_ci.csv       — full CI table
results/tables/bootstrap_ci_summary.csv — key comparisons only

Usage
-----
    python experiments/run_bootstrap_ci.py
    python experiments/run_bootstrap_ci.py --n-bootstrap 2000
    python experiments/run_bootstrap_ci.py --datasets readmissions heart
"""

import argparse
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
from ambiguity_suite import compute_AAPR, compute_AAF1, compute_AAMass

RESULTS_DIR   = PROJECT_ROOT / "results" / "tables"
N_BOOTSTRAP   = 1000
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
DELTA         = 0.10
ALPHA         = 0.05   # 95% CI
DATASETS      = ["heart", "uci_credit", "readmissions", "nslkdd", "brfss", "creditcard"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> tuple:
    """
    Non-parametric bootstrap 95% CI for a scalar metric.

    Parameters
    ----------
    y_true, y_prob : arrays
    metric_fn      : callable(y_true, y_prob) -> float
    n_bootstrap    : number of bootstrap resamples

    Returns
    -------
    (mean, lower_95, upper_95)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_prob[idx]))
    stats = np.array(stats)
    lo = float(np.percentile(stats, 100 * ALPHA / 2))
    hi = float(np.percentile(stats, 100 * (1 - ALPHA / 2)))
    return round(float(stats.mean()), 4), round(lo, 4), round(hi, 4)


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
# Part 1: Synthetic CIs from seed variance
# ---------------------------------------------------------------------------

def synthetic_cis() -> pd.DataFrame:
    """
    Compute mean ± 95% CI for synthetic results using
    seed-level variance (5 seeds × 3 sample sizes = 15 observations).
    """
    syn_path = RESULTS_DIR / "synthetic_results.csv"
    if not syn_path.exists():
        print("  WARNING: synthetic_results.csv not found — skipping synthetic CIs")
        return pd.DataFrame()

    df = pd.read_csv(syn_path)

    # Filter to delta=0.10 for AAMass comparisons
    df_d = df[df["delta"] == DELTA].copy()

    rows = []
    for (mixture, clf), grp in df_d.groupby(["mixture", "classifier"]):
        for metric in ["aapr", "aaf1", "aamass", "auc_roc"]:
            vals = grp[metric].values
            mean = round(float(vals.mean()), 4)
            # 95% CI via percentile over seeds × sample sizes
            lo   = round(float(np.percentile(vals, 2.5)), 4)
            hi   = round(float(np.percentile(vals, 97.5)), 4)
            rows.append({
                "source":     "synthetic",
                "dataset":    mixture,
                "classifier": clf,
                "metric":     metric,
                "mean":       mean,
                "ci_lower":   lo,
                "ci_upper":   hi,
                "ci_width":   round(hi - lo, 4),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2: Real-world bootstrap CIs
# ---------------------------------------------------------------------------

def real_world_cis(
    datasets: list,
    n_bootstrap: int,
) -> pd.DataFrame:
    """
    Fit each classifier once, then bootstrap the test set predictions
    to compute 95% CIs for AAMass, AAPR, AAF1, and AUC.
    """
    rows = []
    classifiers = build_classifiers()

    for dataset_name in datasets:
        print(f"  {dataset_name}...", flush=True)
        try:
            X, y, _ = load_dataset(dataset_name)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y
        )
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        y_test  = np.array(y_test)

        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]

            for metric, fn in [
                ("aamass", lambda yt, yp: compute_AAMass(yt, yp, delta=DELTA)),
                ("aapr",   lambda yt, yp: compute_AAPR(yt, yp)),
                ("aaf1",   lambda yt, yp: compute_AAF1(yt, yp)),
                ("auc_roc",lambda yt, yp: roc_auc_score(yt, yp)),
            ]:
                mean, lo, hi = bootstrap_ci(
                    y_test, y_prob, fn, n_bootstrap=n_bootstrap
                )
                rows.append({
                    "source":     "real",
                    "dataset":    dataset_name,
                    "classifier": clf_name,
                    "metric":     metric,
                    "mean":       mean,
                    "ci_lower":   lo,
                    "ci_upper":   hi,
                    "ci_width":   round(hi - lo, 4),
                })
                print(
                    f"    {clf_name:20s} {metric:8s} "
                    f"{mean:.4f} [{lo:.4f}, {hi:.4f}]",
                    flush=True
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Key comparisons summary
# ---------------------------------------------------------------------------

def key_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the most paper-relevant comparisons."""
    rows = []

    # 1. AUC-matched pair: variant A vs B for LR, AAMass
    syn = df[df["source"] == "synthetic"]
    for variant in ["constant_auc_variant_a", "constant_auc_variant_b"]:
        r = syn[
            (syn["dataset"] == variant) &
            (syn["classifier"] == "LogisticRegression") &
            (syn["metric"] == "aamass")
        ]
        if not r.empty:
            rows.append({
                "comparison": f"Synthetic AUC-matched {variant} LR AAMass",
                "mean": r.iloc[0]["mean"],
                "ci_lower": r.iloc[0]["ci_lower"],
                "ci_upper": r.iloc[0]["ci_upper"],
            })

    # 2. Readmissions: LR vs XGBoost AAMass
    real = df[df["source"] == "real"]
    for clf in ["LogisticRegression", "XGBoost"]:
        r = real[
            (real["dataset"] == "readmissions") &
            (real["classifier"] == clf) &
            (real["metric"] == "aamass")
        ]
        if not r.empty:
            rows.append({
                "comparison": f"Readmissions {clf} AAMass",
                "mean": r.iloc[0]["mean"],
                "ci_lower": r.iloc[0]["ci_lower"],
                "ci_upper": r.iloc[0]["ci_upper"],
            })

    # 3. Heart: RF vs XGBoost AAMass (tied AUC case)
    for clf in ["RandomForest", "XGBoost"]:
        r = real[
            (real["dataset"] == "heart") &
            (real["classifier"] == clf) &
            (real["metric"] == "aamass")
        ]
        if not r.empty:
            rows.append({
                "comparison": f"Heart {clf} AAMass",
                "mean": r.iloc[0]["mean"],
                "ci_lower": r.iloc[0]["ci_lower"],
                "ci_upper": r.iloc[0]["ci_upper"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for ambiguity metrics."
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=N_BOOTSTRAP,
        help=f"Number of bootstrap resamples (default: {N_BOOTSTRAP})"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS,
        help="Real-world datasets to bootstrap (default: all six)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  Bootstrap Confidence Intervals (95%)")
    print(f"  Resamples : {args.n_bootstrap}")
    print(f"  Datasets  : {args.datasets}")
    print(f"  Delta     : {DELTA}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("\n[1/2] Synthetic CIs (seed variance)...")
    syn_df = synthetic_cis()

    print("\n[2/2] Real-world bootstrap CIs...")
    real_df = real_world_cis(args.datasets, args.n_bootstrap)

    all_df = pd.concat([syn_df, real_df], ignore_index=True)

    # Save full CI table
    ci_path = RESULTS_DIR / "bootstrap_ci.csv"
    all_df.to_csv(ci_path, index=False)

    # Save key comparisons
    key_df = key_comparisons(all_df)
    key_path = RESULTS_DIR / "bootstrap_ci_summary.csv"
    key_df.to_csv(key_path, index=False)

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  Full CIs     → {ci_path}")
    print(f"  Key comps    → {key_path}")
    print()
    print("Key comparisons:")
    print(key_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
