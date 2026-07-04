from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def format_float(value: str | float, digits: int = 5) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def make_core_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build a compact Table-I-style summary from completed metric rows."""
    out = []
    for row in rows:
        out.append(
            {
                "Backbone": row.get("backbone", row.get("encoder", "")),
                "Recipe": row.get("recipe", row.get("model", "")),
                "CER": format_float(row.get("cer", "")),
                "Del": format_float(row.get("del", "")),
                "Len": format_float(row.get("len", row.get("len_ratio", ""))),
                "Ins": format_float(row.get("ins", "")),
                "Mis": format_float(row.get("mis", "")),
            }
        )
    return out


def write_markdown_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("No rows available.\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

