#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from basecall_deletion.analysis.tables import make_core_table, read_csv, write_markdown_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate compact markdown tables from completed metric CSV files.")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    rows = make_core_table(read_csv(args.metrics_csv))
    write_markdown_table(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

