"""
experiments/run_calibrated_aamass.py — Calibrated AAMass for all datasets.

Extends the main real-world results table by adding a calibrated AAMass
column. For each classifier and dataset:
    1. Fit classifier on training split
    2. Apply Platt scaling (CalibratedClassifierCV, method='sigmoid')
    3. Compute AAMass on calibrated probabilities
    4. Compare against raw AAMass

This directly addresses the reviewer recommendation:
    "Make calibrated AAMass the main result, raw AAMass the ablation."

The calibration ablation script (run_calibration_ablation.py) already does
this for readmissions and heart. This script extends it to all six datasets
and produces a clean summary table for the paper.

Output
------
results/tables/calibrated_aamass_summary.csv

Usage
-----
    python experiments/run_calibrated_aamass.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import load_dataset
from ambiguity_suite import compute_AAMass

RESULTS_DIR  = PROJECT_ROOT / "results" / "tables"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
DELTA        = 0.10
DATASETS     = ["heart", "uci_credit", "readmissions",
                "nslkdd", "brfss", "creditcard"]


def build_classifiers():
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print("  Calibrated AAMass — All Datasets")
    print(f"  Calibration: Platt scaling (sigmoid)")
    print(f"  Delta: {DELTA}")
    print("=" * 65)

    rows = []

    for dataset_name in DATASETS:
        print(f"\n  {dataset_name}...", flush=True)
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
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        y_te = np.array(y_te)
        prevalence = float(y_te.mean())

        for clf_name, clf in build_classifiers().items():
            import copy

            # Raw
            clf_raw = copy.deepcopy(clf)
            clf_raw.fit(X_tr, y_tr)
            prob_raw = clf_raw.predict_proba(X_te)[:, 1]
            aamass_raw = round(compute_AAMass(y_te, prob_raw, delta=DELTA), 4)
            auc = round(roc_auc_score(y_te, prob_raw), 4)

            # Platt calibrated
            clf_cal = CalibratedClassifierCV(
                copy.deepcopy(clf), method="sigmoid", cv=5
            )
            clf_cal.fit(X_tr, y_tr)
            prob_cal = clf_cal.predict_proba(X_te)[:, 1]
            aamass_cal = round(compute_AAMass(y_te, prob_cal, delta=DELTA), 4)
            auc_cal = round(roc_auc_score(y_te, prob_cal), 4)

            gain = round(aamass_cal - aamass_raw, 4)

            rows.append({
                "dataset":       dataset_name,
                "classifier":    clf_name,
                "prevalence":    round(prevalence, 4),
                "aamass_raw":    aamass_raw,
                "aamass_cal":    aamass_cal,
                "gain":          gain,
                "auc_raw":       auc,
                "auc_cal":       auc_cal,
            })

            print(
                f"    {clf_name:20s}  "
                f"Raw={aamass_raw:.4f}  Cal={aamass_cal:.4f}  "
                f"Gain={gain:+.4f}  AUC={auc:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "calibrated_aamass_summary.csv"
    df.to_csv(out, index=False)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min → {out}")
    print()
    print("Summary:")
    print(df.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
