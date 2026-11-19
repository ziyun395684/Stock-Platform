"""Read-only storage adapter for candidate reports."""
from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from sqlalchemy import select

from .candidate_report import summarize_observations
from .config import settings
from .models import Artifact, Experiment, ExperimentRun
from .repositories import _session_factory


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_artifact(artifact: Artifact) -> list[dict[str, Any]]:
    path = (_root() / artifact.relative_path).resolve()
    root = _root().resolve()
    if root not in path.parents or not path.is_file() or path.stat().st_size > 5_000_000:
        return []
    try:
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".csv":
            return [dict(row) for row in csv.DictReader(StringIO(text))]
        if path.suffix.lower() == ".json":
            payload = json.loads(text)
            return [payload] if isinstance(payload, dict) else []
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    return []


def _observations(run: ExperimentRun, artifacts: list[Artifact]) -> list[dict[str, Any]]:
    base = {"window": run.window_name, "seed": run.seed}
    output: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not any(token in artifact.artifact_type for token in ("oos", "sensitivity", "evaluation", "report", "regime")):
            continue
        for row in _read_artifact(artifact):
            observation = dict(base)
            observation.update(row)
            config = row.get("config") if isinstance(row.get("config"), dict) else {}
            observation.setdefault("window", row.get("training_window") or config.get("training_window") or config.get("window"))
            observation.setdefault("seed", row.get("seed") or config.get("seed"))
            observation.setdefault("cost_rate", row.get("cost_rate") or config.get("cost_rate"))
            observation.setdefault("regime", row.get("regime") or row.get("market_regime"))
            if any(key in observation for key in ("total_return_net", "max_drawdown", "rank_ic", "mean_rank_ic")):
                output.append(observation)
    return output


def build_candidate_report(experiment_id: str) -> dict[str, Any] | None:
    with _session_factory()() as session:
        experiment = session.scalar(select(Experiment).where(Experiment.id == experiment_id))
        if experiment is None:
            return None
        runs = session.scalars(select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id)).all()
        observations: list[dict[str, Any]] = []
        for run in runs:
            artifacts = session.scalars(select(Artifact).where(Artifact.experiment_run_id == run.id)).all()
            observations.extend(_observations(run, artifacts))
        report = summarize_observations(observations, total_runs=len(runs), config=experiment.config_json or {})
        report["experiment_id"] = experiment.experiment_id
        report["experiment_objective"] = experiment.objective
        report["config"] = experiment.config_json or {}
        return report
