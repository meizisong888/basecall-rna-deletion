from __future__ import annotations

from pathlib import Path

from .tables import read_csv


def plot_del_cer_summary(input_csv: str | Path, output_path: str | Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    rows = read_csv(input_csv)
    labels = [row.get("recipe", row.get("model", f"row{i}")) for i, row in enumerate(rows)]
    cer = [float(row.get("cer", 0.0)) for row in rows]
    deletion = [float(row.get("del", row.get("deletion", 0.0))) for row in rows]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(labels)), 3.2))
    x = range(len(labels))
    ax.plot(x, cer, marker="o", label="CER")
    ax.plot(x, deletion, marker="s", label="Del")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Teacher-relative rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

