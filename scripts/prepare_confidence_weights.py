#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from basecall_deletion.training.weights import qw_default


def main() -> int:
    parser = argparse.ArgumentParser(description="Create fixed QW-CTC read weights from reference-alignment confidence.")
    parser.add_argument("--metadata-csv", required=True, help="CSV with read_id and identity columns. Optional split and MAPQ/mapq columns are preserved if present.")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    with Path(args.metadata_csv).open("r", encoding="utf-8", newline="") as src, Path(args.output_csv).open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=["read_id", "split", "identity", "mapq", "weight"])
        writer.writeheader()
        for row in reader:
            read_id = str(row.get("read_id", "")).strip()
            if not read_id:
                continue
            identity = float(row.get("identity", 0.0))
            mapq = row.get("MAPQ", row.get("mapq", ""))
            writer.writerow({"read_id": read_id, "split": row.get("split", ""), "identity": identity, "mapq": mapq, "weight": qw_default(identity)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
