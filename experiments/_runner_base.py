"""
experiments/_runner_base.py — Shared logic for all real-data runners.

Each run_<dataset>.py imports and calls run_dataset() from here.
This keeps the per-dataset scripts minimal and ensures consistent
behaviour across all experiments.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ambiguity_suite import compute_AAPR, compute_AAF1, compute_AAMass

RESULTS_DIR    = PROJECT_ROOT / "results" / "tables"
DEFAULT_DELTAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ECE_N_BINS     = 10   # number of equal-width bins for ECE computation


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
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        ),
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = ECE_N_BINS) -> float:
    """
    Expected Calibration Error (ECE) using equal-width bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    y_true : (n,) int array
    y_prob : (n,) float array of predicted probabilities
    n_bins : number of equal-width bins in [0, 1]

    Returns
    -------
    float in [0, 1]
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # include right edge in last bin
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return round(float(ece), 4)


def run_dataset(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    delta_values: list = None,
    test_size: float = 0.2,
    random_state: int = 42,
    beta: float = 1.0,
) -> pd.DataFrame:
    """
    Train LR, RF, and XGBoost on a real dataset, evaluate all ambiguity
    metrics, calibration metrics (ECE, Brier), and AUC across delta values.

    Parameters
    ----------
    dataset_name : str
    X            : (n, p) float array
    y            : (n,)   int array, values in {0, 1}
    delta_values : list of float
    test_size    : float
    random_state : int
    beta         : float — F-beta weight for AAF1

    Returns
    -------
    pd.DataFrame with columns:
        dataset, classifier, delta,
        aapr, aaf1, aamass,
        auc_roc, brier, ece,
        n_train, n_test, prevalence_train, prevalence_test, fit_time_s
    """
    if delta_values is None:
        delta_values = DEFAULT_DELTAS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{dataset_name}_results.csv"

    n_total    = len(y)
    prevalence = float(y.mean())

    print("=" * 70)
    print(f"Dataset      : {dataset_name}")
    print(f"Samples      : {n_total:,}  |  Features: {X.shape[1]}")
    print(f"Prevalence   : {prevalence:.3f}  ({int(y.sum()):,} positives)")
    print(f"Deltas       : {delta_values}")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    n_train          = len(y_train)
    n_test           = len(y_test)
    prevalence_train = float(y_train.mean())
    prevalence_test  = float(y_test.mean())

    classifiers = build_classifiers(random_state)
    total = len(classifiers) * len(delta_values)
    done  = 0
    rows  = []

    for clf_name, clf in classifiers.items():
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = round(time.perf_counter() - t0, 4)

        y_prob = clf.predict_proba(X_test)[:, 1]

        # --- Standard metrics (delta-invariant) ---
        auc    = round(roc_auc_score(y_test, y_prob), 4)
        brier  = round(brier_score_loss(y_test, y_prob), 4)
        ece    = compute_ece(y_test, y_prob)

        # --- Ambiguity metrics (AAPR/AAF1 delta-invariant) ---
        aapr   = round(compute_AAPR(y_test, y_prob), 4)
        aaf1   = round(compute_AAF1(y_test, y_prob, beta=beta), 4)

        for delta in delta_values:
            aamass = round(compute_AAMass(y_test, y_prob, delta=delta), 4)

            rows.append({
                "dataset":          dataset_name,
                "classifier":       clf_name,
                "delta":            delta,
                "aapr":             aapr,
                "aaf1":             aaf1,
                "aamass":           aamass,
                "auc_roc":          auc,
                "brier":            brier,
                "ece":              ece,
                "n_train":          n_train,
                "n_test":           n_test,
                "prevalence_train": round(prevalence_train, 4),
                "prevalence_test":  round(prevalence_test, 4),
                "fit_time_s":       fit_time,
            })

            done += 1
            pct = done / total * 100
            print(
                f"  [{pct:5.1f}%] {clf_name:20s} | δ={delta:.2f} "
                f"→ AAPR={aapr:.4f}  AAF1={aaf1:.4f}  "
                f"AAMass={aamass:.4f}  "
                f"AUC={auc:.4f}  Brier={brier:.4f}  ECE={ece:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)

    print()
    print("=" * 70)
    print(f"Done. {len(df)} rows written to {results_path}")
    print()

    # Summary — one row per classifier
    summary = (
        df.groupby("classifier")[
            ["aapr", "aaf1", "aamass", "auc_roc", "brier", "ece"]
        ]
        .mean()
        .round(4)
    )
    print("Scores by classifier:")
    print(summary.to_string())
    print("=" * 70)

    return df