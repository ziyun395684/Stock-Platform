"""Factor quantile return decomposition task."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import ROOT, _relative_input, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.factor_stratified_returns")
def factor_stratified_returns(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute daily equal-weight returns by factor quantile and long-short spread."""
    _set_run_status(run_id, "running")
    try:
        import yaml
        import pandas as pd
        from data.panel import load_universe_frames
        from data.panel_pipeline import build_labeled_panel
        from model_and_train.factor_library import add_common_factors
        from research_platform.backend.app.artifact_repository import record_artifact

        raw = dict(config or {})
        data_dir = _relative_input(raw.get("data_dir", "quotes2"), "quotes2")
        config_path = _relative_input(raw.get("config_path", "config.yaml"), "config.yaml")
        with (ROOT / config_path).open(encoding="utf-8") as handle:
            project_config = yaml.safe_load(handle)
        symbols = list(project_config["universe"]["symbols"])
        benchmark_symbol = str(raw.get("benchmark", project_config["universe"].get("benchmark", "510300.SH")))
        start, end = str(raw.get("start", "2024-01-01")), str(raw.get("end", "2024-03-31"))
        hold_days, quantiles = int(raw.get("hold_days", 3)), int(raw.get("quantiles", 5))
        factor = str(raw.get("factor", "momentum_20"))
        frames = load_universe_frames(symbols, data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)
        benchmark = load_universe_frames([benchmark_symbol], data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)[benchmark_symbol]
        panel = build_labeled_panel(frames, benchmark_frame=benchmark, label_threshold=0.0, hold_days=hold_days, date_mode="union", extended_features=True)
        panel = add_common_factors(panel, benchmark_frame=benchmark)
        if factor not in panel.columns:
            raise ValueError(f"factor not found: {factor}")
        panel["future_return"] = panel.groupby("symbol")["close"].shift(-hold_days) / panel["close"] - 1.0
        panel = panel.dropna(subset=[factor, "future_return"]).copy()
        panel["quantile"] = panel.groupby("trade_date")[factor].transform(lambda values: pd.qcut(values.rank(method="first"), quantiles, labels=False, duplicates="drop") + 1)
        daily = panel.groupby(["trade_date", "quantile"], as_index=False)["future_return"].mean().rename(columns={"future_return": "mean_return"})
        pivot = daily.pivot(index="trade_date", columns="quantile", values="mean_return").reset_index()
        qcols = [column for column in pivot.columns if column != "trade_date"]
        if qcols:
            pivot["long_short"] = pivot[max(qcols)] - pivot[min(qcols)]
        summary = daily.groupby("quantile", as_index=False)["mean_return"].agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "mean_return", "std": "volatility", "count": "observations"})
        artifacts = []
        for name, kind, frame in (("factor_stratified_daily.csv", "factor_stratified_daily", pivot), ("factor_stratified_summary.csv", "factor_stratified_summary", summary)):
            path, digest, size = _write_artifact(run_id, name, frame.to_csv(index=False).encode("utf-8-sig"))
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        result = {"task": "factor_stratified_returns", "factor": factor, "quantiles": quantiles, "rows": int(len(daily)), "config": {"start": start, "end": end, "hold_days": hold_days}, "artifacts": artifacts}
        path, digest, size = _write_artifact(run_id, "factor_stratified_metadata.json", json.dumps(result, indent=2).encode("utf-8"))
        artifact_id = record_artifact(run_id, "factor_stratified_metadata", path, digest, size)
        artifacts.append({"id": artifact_id, "artifact_type": "factor_stratified_metadata", "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **result}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
