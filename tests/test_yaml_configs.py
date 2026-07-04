from __future__ import annotations

from pathlib import Path

import yaml


def test_all_configs_parse() -> None:
    for path in Path("configs").glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), path

