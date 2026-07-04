#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from basecall_deletion.analysis.figures import plot_del_cer_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate summary figures from metric CSV files.")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-figure", required=True)
    args = parser.parse_args()
    plot_del_cer_summary(args.metrics_csv, args.output_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

