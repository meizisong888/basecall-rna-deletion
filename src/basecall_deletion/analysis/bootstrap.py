from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import mean
from typing import Any


def read_per_read(path: str | Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            read_id = str(row.get("read_id", "")).strip()
            if not read_id:
                continue
            rows[read_id] = {}
            for key, value in row.items():
                if key == "read_id":
                    continue
                try:
                    rows[read_id][key] = float(value)
                except Exception:
                    pass
    return rows


def bootstrap_ci(values: list[float], n_bootstrap: int = 10000, seed: int = 1) -> tuple[float, float]:
    """Return a paired bootstrap interval for already paired per-read values."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    boots = []
    n = len(values)
    for _ in range(int(n_bootstrap)):
        boots.append(mean(values[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return lo, hi


def one_sided_sign_flip_pvalue(improvements: list[float], n_permutations: int = 200000, seed: int = 1) -> float:
    """One-sided paired sign-flip p-value for positive mean improvement.

    The input values should already be expressed as positive-is-better
    improvements, for example `control_del_count - method_del_count`.
    """
    if not improvements:
        return float("nan")
    rng = random.Random(seed)
    observed = mean(improvements)
    more_extreme = 0
    for _ in range(int(n_permutations)):
        sampled = [value if rng.random() < 0.5 else -value for value in improvements]
        more_extreme += int(mean(sampled) >= observed)
    return (more_extreme + 1) / (int(n_permutations) + 1)


def _metric_value(row: dict[str, float], metric: str) -> float | None:
    aliases = {
        "del": ["del", "deletion", "deletion_rate", "deletion_count"],
        "ins": ["ins", "insertion", "insertion_rate", "insertion_count"],
        "mis": ["mis", "mismatch", "mismatch_rate", "mismatch_count"],
        "len_ratio": ["len_ratio", "len"],
    }
    for key in aliases.get(metric, [metric]):
        if key in row:
            return row[key]
    return None


def paired_metric_rows(
    control_csv: str | Path,
    method_csv: str | Path,
    metrics: list[str],
    n_bootstrap: int = 10000,
    n_permutations: int = 200000,
    seed: int = 1,
    direction: str = "lower",
    relative_del_reduction: bool = False,
) -> list[dict[str, Any]]:
    """Compute paired metric summaries.

    For `direction="lower"`, `mean_improvement` is control minus method, so
    positive values indicate error reduction. For `direction="higher"`, it is
    method minus control. `relative_del_reduction_pct` is computed only for Del
    when requested and is `100 * mean(control_del - method_del) / mean(control_del)`.
    """
    control = read_per_read(control_csv)
    method = read_per_read(method_csv)
    common = sorted(set(control) & set(method))
    rows = []
    for metric in metrics:
        pairs = []
        for read_id in common:
            left = _metric_value(control[read_id], metric)
            right = _metric_value(method[read_id], metric)
            if left is not None and right is not None:
                pairs.append((left, right))
        if direction == "higher":
            improvements = [right - left for left, right in pairs]
            delta_name = "method_minus_control"
        else:
            improvements = [left - right for left, right in pairs]
            delta_name = "control_minus_method"
        lo, hi = bootstrap_ci(improvements, n_bootstrap=n_bootstrap, seed=seed)
        row: dict[str, Any] = {
            "metric": metric,
            "n": len(improvements),
            "delta_definition": delta_name,
            "mean_improvement": mean(improvements) if improvements else "NA",
            "ci_low": lo,
            "ci_high": hi,
            "one_sided_sign_flip_p": one_sided_sign_flip_pvalue(improvements, n_permutations=n_permutations, seed=seed),
        }
        if metric == "del" and relative_del_reduction and pairs:
            baseline_mean = mean([left for left, _right in pairs])
            row["relative_del_reduction_pct"] = 100.0 * mean(improvements) / baseline_mean if baseline_mean else "NA"
        rows.append(row)
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["metric"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
