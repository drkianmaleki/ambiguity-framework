"""
experiments/run_bootstrap_ci_fast.py — Fast bootstrap CIs.

Key differences from run_bootstrap_ci.py:
  - Skips creditcard and brfss (too large, CIs near zero)
  - 500 bootstrap resamples instead of 1000 (still reliable 95% CIs)
  - Bootstraps AAMass and AUC only (AAPR/AAF1 are slow on large datasets)
  - Adds synthetic seed-variance CIs from existing synthetic_results.csv

Estimated runtime: 5–10 minutes.

Usage
-----
    python experiments/run_bootstrap_ci_fast.py
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
N_BOOTSTRAP  = 500
RANDOM_STATE = 42
TEST_SIZE    = 0.2
DELTA        = 0.10

# Skip large datasets — too slow for bootstrap, CIs near zero
DATASETS = ["heart", "uci_credit", "readmissions", "nslkdd"]


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> tuple:
    """95% percentile bootstrap CI for a scalar metric."""
    rng = np.random.default_rng(seed)
    N   = len(y_true)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        stats.append(metric_fn(y_true[idx], y_prob[idx]))
    stats = np.array(stats)
    return (
        round(float(stats.mean()), 4),
        round(float(np.percentile(stats, 2.5)), 4),
        round(float(np.percentile(stats, 97.5)), 4),
    )


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


def real_world_cis() -> pd.DataFrame:
    rows = []

    for dataset_name in DATASETS:
        print(f"\n  {dataset_name}...", flush=True)
        X, y, _ = load_dataset(dataset_name)

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

            for metric, fn in [
                ("aamass",  lambda yt, yp: compute_AAMass(yt, yp, delta=DELTA)),
                ("auc_roc", lambda yt, yp: roc_auc_score(yt, yp)),
            ]:
                mean, lo, hi = bootstrap_ci(y_te, y_prob, fn)
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
                    f"    {clf_name:20s} {metric:8s}  "
                    f"{mean:.4f} [{lo:.4f}, {hi:.4f}]",
                    flush=True,
                )

    return pd.DataFrame(rows)


def synthetic_cis() -> pd.DataFrame:
    """Compute mean ± 95% CI from seed variance in synthetic_results.csv."""
    syn_path = RESULTS_DIR / "synthetic_results.csv"
    if not syn_path.exists():
        print("  synthetic_results.csv not found — skipping")
        return pd.DataFrame()

    df  = pd.read_csv(syn_path)
    df  = df[df["delta"] == DELTA]
    rows = []

    for (mixture, clf), grp in df.groupby(["mixture", "classifier"]):
        for metric in ["aapr", "aaf1", "aamass", "auc_roc"]:
            vals = grp[metric].values
            rows.append({
                "source":     "synthetic",
                "dataset":    mixture,
                "classifier": clf,
                "metric":     metric,
                "mean":       round(float(vals.mean()), 4),
                "ci_lower":   round(float(np.percentile(vals, 2.5)), 4),
                "ci_upper":   round(float(np.percentile(vals, 97.5)), 4),
                "ci_width":   round(float(np.percentile(vals, 97.5)) -
                                    float(np.percentile(vals, 2.5)), 4),
            })

    return pd.DataFrame(rows)


def print_key_comparisons(df: pd.DataFrame) -> None:
    print("\nKey comparisons for paper:")
    print("-" * 65)

    # 1. Heart: RF vs XGBoost AAMass (tied AUC case)
    real = df[df["source"] == "real"]
    print("\n  Heart Disease — RF vs XGBoost (AUC tied at 0.923):")
    for clf in ["RandomForest", "XGBoost"]:
        r = real[(real["dataset"]=="heart") & (real["classifier"]==clf) &
                 (real["metric"]=="aamass")]
        if not r.empty:
            row = r.iloc[0]
            print(f"    {clf:20s} AAMass={row['mean']:.4f} "
                  f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

    # 2. Readmissions: LR vs XGBoost AAMass
    print("\n  Readmissions — LR vs XGBoost:")
    for clf in ["LogisticRegression", "XGBoost"]:
        r = real[(real["dataset"]=="readmissions") & (real["classifier"]==clf) &
                 (real["metric"]=="aamass")]
        if not r.empty:
            row = r.iloc[0]
            print(f"    {clf:20s} AAMass={row['mean']:.4f} "
                  f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

    # 3. AUC-matched pair from synthetic
    syn = df[df["source"] == "synthetic"]
    print("\n  Synthetic AUC-matched pair — LR AAMass:")
    for variant in ["constant_auc_variant_a", "constant_auc_variant_b"]:
        r = syn[(syn["dataset"]==variant) & (syn["classifier"]=="LogisticRegression") &
                (syn["metric"]=="aamass")]
        if not r.empty:
            row = r.iloc[0]
            print(f"    {variant:35s} {row['mean']:.4f} "
                  f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

    print("-" * 65)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print(f"  Fast Bootstrap CIs  ({N_BOOTSTRAP} resamples, δ={DELTA})")
    print(f"  Datasets : {DATASETS}")
    print("=" * 65)

    print("\n[1/2] Real-world bootstrap CIs...")
    real_df = real_world_cis()

    print("\n[2/2] Synthetic CIs (seed variance)...")
    syn_df = synthetic_cis()

    all_df = pd.concat([real_df, syn_df], ignore_index=True)

    out = RESULTS_DIR / "bootstrap_ci_fast.csv"
    all_df.to_csv(out, index=False)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min → {out}")

    print_key_comparisons(all_df)


if __name__ == "__main__":
    main()
