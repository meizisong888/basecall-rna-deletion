#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from basecall_deletion.training.train_loop import train_from_config
from basecall_deletion.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a QW-CTC student model from pre-materialized CTC chunks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--pseudo-label-fastq", default="")
    parser.add_argument("--read-weight-table", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("data", {})
    if args.data_root:
        config["data"]["data_root"] = args.data_root
    if args.read_weight_table:
        config["data"]["read_weight_table"] = args.read_weight_table
    if args.pseudo_label_fastq:
        config["data"]["pseudo_label_fastq"] = args.pseudo_label_fastq
    result = train_from_config(config, args.output_dir)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

