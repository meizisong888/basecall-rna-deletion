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


def sign_flip_pvalue(deltas: list[float], n_permutations: int = 10000, seed: int = 1) -> float:
    if not deltas:
        return float("nan")
    rng = random.Random(seed)
    observed = abs(mean(deltas))
    more_extreme = 0
    for _ in range(int(n_permutations)):
        sampled = [delta if rng.random() < 0.5 else -delta for delta in deltas]
        more_extreme += int(abs(mean(sampled)) >= observed)
    return (more_extreme + 1) / (int(n_permutations) + 1)


def paired_metric_rows(control_csv: str | Path, method_csv: str | Path, metrics: list[str], n_bootstrap: int = 10000, seed: int = 1) -> list[dict[str, Any]]:
    control = read_per_read(control_csv)
    method = read_per_read(method_csv)
    common = sorted(set(control) & set(method))
    rows = []
    for metric in metrics:
        deltas = [method[read_id][metric] - control[read_id][metric] for read_id in common if metric in control[read_id] and metric in method[read_id]]
        lo, hi = bootstrap_ci(deltas, n_bootstrap=n_bootstrap, seed=seed)
        rows.append(
            {
                "metric": metric,
                "n": len(deltas),
                "mean_delta": mean(deltas) if deltas else "NA",
                "ci_low": lo,
                "ci_high": hi,
                "sign_flip_p": sign_flip_pvalue(deltas, seed=seed),
            }
        )
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["metric"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

