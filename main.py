"""
main.py — Command-line entry point for the Ambiguity Range Framework.

Provides a quick way to compute ambiguity metrics on any CSV dataset
without writing Python code.

Usage
-----
    python main.py --help

    # Compute all three metrics on a CSV file
    python main.py \\
        --data path/to/predictions.csv \\
        --y-true label \\
        --y-prob score \\
        --delta 0.10

    # Example with multiple delta values
    python main.py \\
        --data results/tables/heart_results.csv \\
        --y-true y_true \\
        --y-prob y_prob \\
        --delta 0.05 0.10 0.20

Input CSV format
----------------
The CSV must contain at least two columns:
  - A ground-truth label column (0 or 1)
  - A predicted probability column (float in [0, 1])

Output
------
Prints a summary table of AAPR, AAF1, and AAMass to stdout.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from ambiguity_suite import compute_AAPR, compute_AAF1, compute_AAMass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Ambiguity Range Framework metrics on a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data",    required=True,  help="Path to CSV file")
    parser.add_argument("--y-true",  required=True,  help="Column name for ground-truth labels")
    parser.add_argument("--y-prob",  required=True,  help="Column name for predicted probabilities")
    parser.add_argument(
        "--delta", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        help="Delta values for AA_Mass (default: 0.05 0.10 0.15 0.20 0.25 0.30)",
    )
    parser.add_argument("--beta", type=float, default=1.0,
                        help="F-beta weight for AAF1 (default: 1.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        print(f"Error: file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    for col in [args.y_true, args.y_prob]:
        if col not in df.columns:
            print(f"Error: column '{col}' not found in {args.data}", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    y_true = df[args.y_true].values.astype(int)
    y_prob = df[args.y_prob].values.astype(float)

    prevalence = y_true.mean()
    print(f"\nDataset  : {args.data}")
    print(f"Samples  : {len(y_true):,}")
    print(f"Prevalence: {prevalence:.3f}")
    print()

    aapr = compute_AAPR(y_true, y_prob)
    aaf1 = compute_AAF1(y_true, y_prob, beta=args.beta)

    print(f"  AA_PR   = {aapr:.4f}  (global, delta-invariant)")
    print(f"  AA_F1   = {aaf1:.4f}  (global, delta-invariant, beta={args.beta})")
    print()
    print(f"  {'delta':>8}  {'AA_Mass':>10}")
    print(f"  {'-'*8}  {'-'*10}")
    for delta in args.delta:
        aamass = compute_AAMass(y_true, y_prob, delta=delta)
        print(f"  {delta:>8.2f}  {aamass:>10.4f}")
    print()


if __name__ == "__main__":
    main()
