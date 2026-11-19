"""Shared, bounded metric extraction for research run artifacts."""
from __future__ import annotations

import csv
import json
import math
from io import StringIO
from pathlib import Path
from typing import Any

METRICS = ("total_return_net", "max_drawdown", "sharpe", "turnover", "rank_ic", "mean_rank_ic", "positive_ratio")
MAX_ARTIFACT_BYTES = 2_000_000


def artifact_root() -> Path:
    root = Path("/workspace") if Path("/workspace").exists() else Path.cwd()
    return root.resolve()


def collect_metrics(value: Any, output: dict[str, list[float]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in METRICS and isinstance(item, (int, float)) and math.isfinite(float(item)):
                output.setdefault(key, []).append(float(item))
            collect_metrics(item, output)
    elif isinstance(value, list):
        for item in value:
            collect_metrics(item, output)


def read_run_metrics(run: dict[str, Any], root: Path | None = None) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    root = (root or artifact_root()).resolve()
    for artifact in run.get("artifacts", []):
        relative_path = artifact.get("relative_path")
        if not relative_path:
            continue
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
            if path.suffix.lower() == ".json":
                collect_metrics(json.loads(text), output)
            elif path.suffix.lower() == ".csv":
                for row in csv.DictReader(StringIO(text)):
                    for name in METRICS:
                        value = row.get(name)
                        if value in (None, ""):
                            continue
                        try:
                            number = float(value)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(number):
                            output.setdefault(name, []).append(number)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return output


def average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 8) if values else None


def average_metrics(metric_values: dict[str, list[float]]) -> dict[str, float | None]:
    return {name: average(values) for name, values in metric_values.items()}

