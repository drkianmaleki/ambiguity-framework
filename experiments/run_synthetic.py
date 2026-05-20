"""
run_synthetic.py — Synthetic experiment runner.

Generates labelled datasets from Gaussian mixtures defined in
experiments/configs/experiment_grid.yaml, trains three classifiers
(Logistic Regression, Random Forest, XGBoost), evaluates all three
ambiguity metrics (AAPR, AAF1, AAMass) across a range of delta values,
and writes results to results/tables/synthetic_results.csv.

Usage
-----
    python experiments/run_synthetic.py                      # all mixtures
    python experiments/run_synthetic.py --mixture well_separated
    python experiments/run_synthetic.py --delta 0.05 0.10 0.20

Output columns
--------------
mixture, n_samples, seed, classifier, delta,
aapr, aaf1, aamass, auc_roc, fit_time_s
"""

import argparse
import time
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or experiments/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

# If this script lives at <root>/experiments/run_synthetic.py → parent is root
# If it lives at <root>/run_synthetic.py (dev convenience) → parent is root too
if _HERE.name == "experiments":
    PROJECT_ROOT = _HERE.parent
else:
    PROJECT_ROOT = _HERE

sys.path.insert(0, str(PROJECT_ROOT))

from ambiguity_suite import compute_AAPR, compute_AAF1, compute_AAMass

CONFIG_PATH  = PROJECT_ROOT / "experiments" / "configs" / "experiment_grid.yaml"
RESULTS_DIR  = PROJECT_ROOT / "results" / "tables"
RESULTS_PATH = RESULTS_DIR / "synthetic_results.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_gaussian_mixture(
    mu: list,
    sigma: list,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced binary dataset from two 2-D Gaussian clusters.

    Parameters
    ----------
    mu     : [[mu0_x, mu0_y], [mu1_x, mu1_y]]
    sigma  : [[s0_x, s0_y], [s1_x, s1_y]]  (diagonal covariance entries)
    n_samples : total samples (split equally between classes)
    seed   : random seed

    Returns
    -------
    X : (n_samples, 2) float array
    y : (n_samples,)  int array  {0, 1}
    """
    rng  = np.random.default_rng(seed)
    half = n_samples // 2

    mu0, mu1       = np.array(mu[0]),    np.array(mu[1])
    sig0, sig1     = np.array(sigma[0]), np.array(sigma[1])

    X0 = rng.normal(loc=mu0, scale=sig0, size=(half, 2))
    X1 = rng.normal(loc=mu1, scale=sig1, size=(half, 2))

    X = np.vstack([X0, X1])
    y = np.array([0] * half + [1] * half)
    return X, y


def build_classifiers(random_state: int) -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: dict,
    delta_values: list[float],
    mixture_filter: str | None,
) -> pd.DataFrame:
    """
    Run all combinations of mixture × n_samples × seed × classifier × delta.

    Returns a DataFrame with one row per combination.
    """
    syn_cfg  = cfg["synthetic"]
    exp_cfg  = cfg["experiments"]
    mixtures = syn_cfg["gaussian_mixtures"]

    if mixture_filter:
        mixtures = [m for m in mixtures if m["name"] == mixture_filter]
        if not mixtures:
            raise ValueError(
                f"Mixture '{mixture_filter}' not found in config. "
                f"Available: {[m['name'] for m in syn_cfg['gaussian_mixtures']]}"
            )

    test_size    = exp_cfg["test_size"]
    random_state = exp_cfg["random_state"]
    beta         = exp_cfg["metrics"]["aaf1"]["beta"]

    rows = []
    total = (
        len(mixtures)
        * len(syn_cfg["n_samples"])
        * len(syn_cfg["seeds"])
        * len(build_classifiers(random_state))
        * len(delta_values)
    )
    done = 0

    for mixture in mixtures:
        name  = mixture["name"]
        mu    = mixture["mu"]
        sigma = mixture["sigma"]

        for n_samples, seed in product(syn_cfg["n_samples"], syn_cfg["seeds"]):
            X, y = make_gaussian_mixture(mu, sigma, n_samples, seed)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            classifiers = build_classifiers(random_state)

            for clf_name, clf in classifiers.items():
                t0 = time.perf_counter()
                clf.fit(X_train, y_train)
                fit_time = round(time.perf_counter() - t0, 4)

                y_prob = clf.predict_proba(X_test)[:, 1]
                auc    = round(roc_auc_score(y_test, y_prob), 4)

                for delta in delta_values:
                    aapr   = round(compute_AAPR(y_test, y_prob), 4)
                    aaf1   = round(compute_AAF1(y_test, y_prob, beta=beta), 4)
                    aamass = round(compute_AAMass(y_test, y_prob, delta=delta), 4)

                    rows.append({
                        "mixture":    name,
                        "n_samples":  n_samples,
                        "seed":       seed,
                        "classifier": clf_name,
                        "delta":      delta,
                        "aapr":       aapr,
                        "aaf1":       aaf1,
                        "aamass":     aamass,
                        "auc_roc":    auc,
                        "fit_time_s": fit_time,
                    })

                    done += 1
                    pct = done / total * 100
                    print(
                        f"  [{pct:5.1f}%] {name:25s} | n={n_samples:5d} "
                        f"| seed={seed:4d} | {clf_name:20s} | δ={delta:.2f} "
                        f"→ AAPR={aapr:.4f}  AAF1={aaf1:.4f}  "
                        f"AAMass={aamass:.4f}  AUC={auc:.4f}",
                        flush=True,
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic ambiguity experiments."
    )
    parser.add_argument(
        "--mixture",
        type=str,
        default=None,
        help="Run a single named mixture (default: all)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        help="Delta values to evaluate (default: 0.05 0.10 0.15 0.20 0.25 0.30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(CONFIG_PATH)

    print("=" * 70)
    print("Synthetic ambiguity experiment")
    print(f"  Config  : {CONFIG_PATH}")
    print(f"  Deltas  : {args.delta}")
    print(f"  Mixture : {args.mixture or 'all'}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = run_experiment(cfg, args.delta, args.mixture)

    df.to_csv(RESULTS_PATH, index=False)

    print()
    print("=" * 70)
    print(f"Done. {len(df)} rows written to {RESULTS_PATH}")
    print()

    # --- Summary table ---
    summary = (
        df.groupby(["mixture", "classifier"])[["aapr", "aaf1", "aamass", "auc_roc"]]
        .mean()
        .round(4)
    )
    print("Mean scores across seeds and n_samples:")
    print(summary.to_string())
    print("=" * 70)

    summary_path = RESULTS_DIR / "synthetic_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
