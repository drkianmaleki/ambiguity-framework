"""
run_brfss.py — Ambiguity experiment on the BRFSS 2015 dataset.

Dataset : heart_disease_health_indicators_BRFSS2015.csv
          (253,680 rows, 21 features, 9.0% positive)

Usage
-----
    python experiments/run_brfss.py
    python experiments/run_brfss.py --delta 0.05 0.10 0.20
    python experiments/run_brfss.py --force-reload
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

    X, y, _ = load_dataset("brfss", force_reload=args.force_reload)
    run_dataset("brfss", X, y, delta_values=args.delta)


if __name__ == "__main__":
    main()
