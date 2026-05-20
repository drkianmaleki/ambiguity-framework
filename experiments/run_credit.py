"""
run_credit.py — Ambiguity experiment on the Credit Card Fraud dataset.

Dataset : creditcard.csv  (284,807 rows, 30 features, 0.17% positive)
Note    : Highly imbalanced — stratified split preserves class ratio.

Usage
-----
    python experiments/run_credit.py
    python experiments/run_credit.py --delta 0.05 0.10 0.20
    python experiments/run_credit.py --force-reload
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import load_dataset
from experiments._runner_base import run_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()

    X, y, _ = load_dataset("creditcard", force_reload=args.force_reload)
    run_dataset("creditcard", X, y, delta_values=args.delta)


if __name__ == "__main__":
    main()
