"""
run_heart.py — Ambiguity experiment on the Heart Disease dataset.

Dataset : heart.csv  (918 rows, 11 features, 55.3% positive)

Usage
-----
    python experiments/run_heart.py
    python experiments/run_heart.py --delta 0.05 0.10 0.20
    python experiments/run_heart.py --force-reload
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

    X, y, _ = load_dataset("heart", force_reload=args.force_reload)
    run_dataset("heart", X, y, delta_values=args.delta)


if __name__ == "__main__":
    main()
