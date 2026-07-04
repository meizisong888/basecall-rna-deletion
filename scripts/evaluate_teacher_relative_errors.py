#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from basecall_deletion.evaluation.teacher_relative import evaluate_fastq


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate decoded student sequences against fixed teacher pseudo-labels.")
    parser.add_argument("--pred-fastq", required=True)
    parser.add_argument("--teacher-fastq", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = evaluate_fastq(args.pred_fastq, args.teacher_fastq, args.output_dir)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

