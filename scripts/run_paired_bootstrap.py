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
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rows = paired_metric_rows(
        args.control_per_read,
        args.method_per_read,
        [part.strip() for part in args.metrics.split(",") if part.strip()],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    write_rows(args.output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

