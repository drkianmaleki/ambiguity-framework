"""
run_nslkdd.py — Ambiguity experiment on the NSL-KDD dataset.

Dataset : nsl_kdd_dataset.csv  (125,973 rows, 41 features, 46.5% positive)
Note    : Binary framing — normal=0, all attack classes=1.

Usage
-----
    python experiments/run_nslkdd.py
    python experiments/run_nslkdd.py --delta 0.05 0.10 0.20
    python experiments/run_nslkdd.py --force-reload
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

    X, y, _ = load_dataset("nslkdd", force_reload=args.force_reload)
    run_dataset("nslkdd", X, y, delta_values=args.delta)


if __name__ == "__main__":
    main()
