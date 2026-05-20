"""
run_uci_credit.py — Ambiguity experiment on the UCI Credit Card dataset.

Dataset : UCI_Credit_Card.csv  (30,000 rows, 23 features, 22.1% positive)

Usage
-----
    python experiments/run_uci_credit.py
    python experiments/run_uci_credit.py --delta 0.05 0.10 0.20
    python experiments/run_uci_credit.py --force-reload
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

    X, y, _ = load_dataset("uci_credit", force_reload=args.force_reload)
    run_dataset("uci_credit", X, y, delta_values=args.delta)


if __name__ == "__main__":
    main()
