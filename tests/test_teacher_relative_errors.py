from __future__ import annotations

from basecall_deletion.evaluation.teacher_relative import evaluate_records


def metric(pred: str, teacher: str) -> dict[str, object]:
    rows, _summary = evaluate_records({"r1": pred}, {"r1": teacher})
    return rows[0]


def test_missing_teacher_base_counts_as_deletion() -> None:
    row = metric("ACT", "ACGT")
    assert row["del"] == 0.25
    assert row["ins"] == 0
    assert row["mis"] == 0


def test_extra_predicted_base_counts_as_insertion() -> None:
    row = metric("ACGT", "ACT")
    assert row["ins"] == 1 / 3
    assert row["del"] == 0
    assert row["mis"] == 0


def test_substituted_base_counts_as_mismatch() -> None:
    row = metric("ACCT", "ACGT")
    assert row["mis"] == 0.25
    assert row["del"] == 0
    assert row["ins"] == 0

