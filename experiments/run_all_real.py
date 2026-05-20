"""
experiments/run_all_real.py — Run all six real-world dataset experiments.

Runs each dataset sequentially, saves individual CSVs to results/tables/,
and writes a combined summary to results/tables/real_summary.csv.

Usage
-----
    python experiments/run_all_real.py
    python experiments/run_all_real.py --delta 0.05 0.10 0.20
    python experiments/run_all_real.py --force-reload

Estimated runtime (on a modern laptop, no GPU):
    heart           ~5 sec
    uci_credit      ~30 sec
    readmissions    ~45 sec
    nslkdd          ~10 sec
    brfss           ~5 min
    creditcard      ~15 min
    --------------------------------
    Total           ~20-25 min
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_loader import load_dataset
from experiments._runner_base import run_dataset, RESULTS_DIR

import pandas as pd

DATASETS = [
    "heart",
    "uci_credit",
    "readmissions",
    "nslkdd",
    "brfss",
    "creditcard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all real-world ambiguity experiments."
    )
    parser.add_argument(
        "--delta", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        help="Delta values to evaluate (default: 0.05 0.10 0.15 0.20 0.25 0.30)",
    )
    parser.add_argument(
        "--force-reload", action="store_true",
        help="Bypass the dataset cache and re-process from raw CSV",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS,
        help="Subset of datasets to run (default: all six)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wall_start = time.time()
    results = []
    failed  = []

    print("\n" + "=" * 70)
    print("  Ambiguity Framework — Real-World Experiment Suite")
    print(f"  Datasets : {args.datasets}")
    print(f"  Deltas   : {args.delta}")
    print(f"  Reload   : {args.force_reload}")
    print("=" * 70 + "\n")

    for i, name in enumerate(args.datasets, 1):
        print(f"\n[{i}/{len(args.datasets)}] Starting: {name}")
        t0 = time.time()
        try:
            X, y, _ = load_dataset(name, force_reload=args.force_reload)
            df = run_dataset(name, X, y, delta_values=args.delta)
            results.append(df)
            elapsed = time.time() - t0
            print(f"  Finished {name} in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  ERROR on {name} after {elapsed:.1f}s: {e}")
            failed.append((name, str(e)))

    # --- Combined summary ---
    if results:
        combined = pd.concat(results, ignore_index=True)

        summary = (
            combined.groupby(["dataset", "classifier"])[
                ["aapr", "aaf1", "aamass", "auc_roc", "brier", "ece"]
            ]
            .mean()
            .round(4)
        )

        summary_path = RESULTS_DIR / "real_summary.csv"
        summary.to_csv(summary_path)

        total_elapsed = time.time() - wall_start
        print("\n" + "=" * 70)
        print(f"  All datasets complete in {total_elapsed/60:.1f} min")
        print(f"  Summary saved to {summary_path}")
        print()
        print(summary.to_string())
        print("=" * 70)

    if failed:
        print("\nFailed datasets:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()