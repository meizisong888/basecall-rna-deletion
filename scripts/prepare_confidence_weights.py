#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def qw_default(identity: float) -> float:
    if identity >= 0.97:
        return 1.0
    if identity >= 0.95:
        return 0.8
    if identity >= 0.90:
        return 0.5
    return 0.2


def main() -> int:
    parser = argparse.ArgumentParser(description="Create fixed QW-CTC read weights from reference-alignment confidence.")
    parser.add_argument("--metadata-csv", required=True, help="CSV with read_id, split, and identity columns.")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    with Path(args.metadata_csv).open("r", encoding="utf-8", newline="") as src, Path(args.output_csv).open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=["read_id", "split", "identity", "weight"])
        writer.writeheader()
        for row in reader:
            read_id = str(row.get("read_id", "")).strip()
            if not read_id:
                continue
            identity = float(row.get("identity", 0.0))
            writer.writerow({"read_id": read_id, "split": row.get("split", ""), "identity": identity, "weight": qw_default(identity)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

