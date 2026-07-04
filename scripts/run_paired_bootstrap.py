#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from basecall_deletion.analysis.bootstrap import paired_metric_rows, write_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired bootstrap and sign-flip tests on per-read metrics.")
    parser.add_argument("--control-per-read", required=True)
    parser.add_argument("--method-per-read", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metrics", default="cer,del,ins,mis,len_ratio")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=200000)
    parser.add_argument("--direction", choices=["lower", "higher"], default="lower")
    parser.add_argument("--relative-del-reduction", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use 1000 bootstrap samples and 1000 sign flips for smoke testing.")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    n_bootstrap = 1000 if args.quick else args.n_bootstrap
    n_permutations = 1000 if args.quick else args.n_permutations
    rows = paired_metric_rows(
        args.control_per_read,
        args.method_per_read,
        [part.strip() for part in args.metrics.split(",") if part.strip()],
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        direction=args.direction,
        relative_del_reduction=args.relative_del_reduction,
        seed=args.seed,
    )
    write_rows(args.output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
