from __future__ import annotations

import csv
import json
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Iterable

from basecall_deletion.data.io import read_fastq

GAP = "-"


def canonical(sequence: str | None) -> str:
    return "".join(base for base in str(sequence or "").upper().replace("U", "T") if base in "ACGT")


def sequence_matcher_pairs(pred: str, teacher: str) -> list[tuple[str, str]]:
    pred = canonical(pred)
    teacher = canonical(teacher)
    pairs: list[tuple[str, str]] = []
    matcher = SequenceMatcher(a=pred, b=teacher, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend(zip(pred[i1:i2], teacher[j1:j2]))
        elif tag == "replace":
            left = pred[i1:i2]
            right = teacher[j1:j2]
            common = min(len(left), len(right))
            pairs.extend(zip(left[:common], right[:common]))
            pairs.extend((base, GAP) for base in left[common:])
            pairs.extend((GAP, base) for base in right[common:])
        elif tag == "delete":
            pairs.extend((base, GAP) for base in pred[i1:i2])
        elif tag == "insert":
            pairs.extend((GAP, base) for base in teacher[j1:j2])
    return pairs


def align_pairs(pred: str, teacher: str) -> list[tuple[str, str]]:
    try:
        import edlib

        alignment = edlib.align(canonical(pred), canonical(teacher), task="path")
        cigar = str(alignment.get("cigar") or "")
        if cigar:
            return pairs_from_cigar(canonical(pred), canonical(teacher), cigar)
    except Exception:
        pass
    return sequence_matcher_pairs(pred, teacher)


def pairs_from_cigar(pred: str, teacher: str, cigar: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pred_pos = 0
    teacher_pos = 0
    number = ""
    for char in cigar:
        if char.isdigit():
            number += char
            continue
        count = int(number or "1")
        number = ""
        if char in {"=", "X", "M"}:
            for _ in range(count):
                pairs.append((pred[pred_pos], teacher[teacher_pos]))
                pred_pos += 1
                teacher_pos += 1
        elif char == "I":
            for _ in range(count):
                pairs.append((pred[pred_pos], GAP))
                pred_pos += 1
        elif char == "D":
            for _ in range(count):
                pairs.append((GAP, teacher[teacher_pos]))
                teacher_pos += 1
    return pairs


def count_ops(pairs: Iterable[tuple[str, str]]) -> dict[str, int]:
    counts = {"matches": 0, "mismatch_count": 0, "insertion_count": 0, "deletion_count": 0}
    for pred, teacher in pairs:
        if pred == GAP and teacher != GAP:
            counts["deletion_count"] += 1
        elif pred != GAP and teacher == GAP:
            counts["insertion_count"] += 1
        elif pred == teacher:
            counts["matches"] += 1
        else:
            counts["mismatch_count"] += 1
    counts["edit_distance"] = counts["mismatch_count"] + counts["insertion_count"] + counts["deletion_count"]
    return counts


def evaluate_records(predictions: dict[str, str], teacher: dict[str, str]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for read_id, teacher_seq in teacher.items():
        pred_seq = predictions.get(read_id, "")
        pairs = align_pairs(pred_seq, teacher_seq)
        counts = count_ops(pairs)
        teacher_len = len(canonical(teacher_seq))
        denom = max(teacher_len, 1)
        rows.append(
            {
                "read_id": read_id,
                "teacher_len": teacher_len,
                "pred_len": len(canonical(pred_seq)),
                "cer": counts["edit_distance"] / denom,
                "del": counts["deletion_count"] / denom,
                "ins": counts["insertion_count"] / denom,
                "mis": counts["mismatch_count"] / denom,
                "len_ratio": len(canonical(pred_seq)) / denom,
                **counts,
            }
        )
    summary = {
        "num_reads": len(rows),
        "cer": mean([float(row["cer"]) for row in rows]) if rows else 0.0,
        "del": mean([float(row["del"]) for row in rows]) if rows else 0.0,
        "ins": mean([float(row["ins"]) for row in rows]) if rows else 0.0,
        "mis": mean([float(row["mis"]) for row in rows]) if rows else 0.0,
        "len": mean([float(row["len_ratio"]) for row in rows]) if rows else 0.0,
    }
    return rows, summary


def evaluate_fastq(pred_fastq: str | Path, teacher_fastq: str | Path, output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate_records(read_fastq(pred_fastq), read_fastq(teacher_fastq))
    with (output / "per_read_teacher_relative_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["read_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary_teacher_relative_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

