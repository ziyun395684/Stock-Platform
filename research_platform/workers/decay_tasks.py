"""Celery adapter for multi-horizon factor IC decay analysis."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import ROOT, _relative_input, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.factor_decay")
def factor_decay(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Measure cross-sectional Rank IC over configurable forward horizons."""
    _set_run_status(run_id, "running")
    try:
        import numpy as np
        import pandas as pd
        import yaml
        from data.panel import align_etf_frames, load_universe_frames
        from model_and_train.factor_library import add_common_factors
        from research_platform.backend.app.artifact_repository import record_artifact

        raw = dict(config or {})
        config_path = _relative_input(raw.get("config_path", "config.yaml"), "config.yaml")
        data_dir = _relative_input(raw.get("data_dir", "quotes2"), "quotes2")
        with (ROOT / config_path).open(encoding="utf-8") as handle:
            project_config = yaml.safe_load(handle)
        symbols = list(project_config["universe"]["symbols"])
        start = str(raw.get("start", "2024-01-01"))
        end = str(raw.get("end", "2024-03-31"))
        horizons = [int(value) for value in (raw.get("horizons") or [1, 3, 5, 10, 20])]
        if any(value < 1 for value in horizons):
            raise ValueError("horizons 必须为正整数")
        frames = load_universe_frames(symbols, data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)
        panel = add_common_factors(align_etf_frames(frames, date_mode="union"))
        factors = raw.get("factor_columns") or ["roc_5", "roc_10", "roc_20", "trend_score_24", "ma_energy_41", "volume_ratio_5_20"]
        factors = [factor for factor in factors if factor in panel.columns]
        if not factors:
            raise ValueError("没有可用于衰减分析的因子列")
        panel = panel.sort_values(["symbol", "trade_date"]).copy()
        grouped = panel.groupby("symbol", sort=False)
        rows: list[dict[str, Any]] = []
        for horizon in horizons:
            panel[f"forward_return_{horizon}"] = grouped["close"].shift(-horizon) / panel["close"] - 1.0
            for factor in factors:
                for date, group in panel.groupby("trade_date", sort=True):
                    clean = group[[factor, f"forward_return_{horizon}"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if len(clean) < 3 or clean[factor].nunique() < 2 or clean[f"forward_return_{horizon}"].nunique() < 2:
                        continue
                    ic = clean[factor].corr(clean[f"forward_return_{horizon}"], method="spearman")
                    if pd.notna(ic):
                        rows.append({"trade_date": int(date), "horizon": horizon, "factor": factor, "rank_ic": float(ic), "n_symbols": int(len(clean))})
        daily = pd.DataFrame(rows)
        if daily.empty:
            raise RuntimeError("没有生成因子衰减日度统计")
        summary = daily.groupby(["horizon", "factor"], as_index=False).agg(n_dates=("trade_date", "nunique"), mean_rank_ic=("rank_ic", "mean"), median_rank_ic=("rank_ic", "median"), positive_ic_ratio=("rank_ic", lambda values: float((values > 0).mean())))
        artifacts = []
        for name, kind, content in (("factor_decay_summary.csv", "factor_decay_summary", summary.to_csv(index=False).encode("utf-8-sig")), ("factor_decay_daily.csv", "factor_decay_daily", daily.to_csv(index=False).encode("utf-8-sig"))):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        metadata = {"task": "factor_decay", "horizons": horizons, "factor_columns": factors, "rows": int(len(daily)), "start": start, "end": end, "data_dir": str(data_dir)}
        path, digest, size = _write_artifact(run_id, "factor_decay_summary.json", json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))
        artifact_id = record_artifact(run_id, "factor_decay_metadata", path, digest, size)
        artifacts.append({"id": artifact_id, "artifact_type": "factor_decay_metadata", "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **metadata, "artifacts": artifacts}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
