"""Celery adapter for regime-conditioned factor Rank IC analysis."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import ROOT, _relative_input, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.factor_regime")
def factor_regime(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify benchmark regimes and summarize factor Rank IC within each regime."""
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
        benchmark_symbol = str(raw.get("benchmark", project_config["universe"].get("benchmark", "510300.SH")))
        start, end = str(raw.get("start", "2024-01-01")), str(raw.get("end", "2024-03-31"))
        frames = load_universe_frames(symbols, data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)
        benchmark = load_universe_frames([benchmark_symbol], data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)[benchmark_symbol]
        panel = add_common_factors(align_etf_frames(frames, date_mode="union"))
        benchmark = benchmark.sort_index().copy()
        benchmark["benchmark_return_1"] = benchmark["close"].pct_change()
        benchmark["trend_return_60"] = benchmark["close"].pct_change(60)
        benchmark["volatility_20"] = benchmark["benchmark_return_1"].rolling(20).std()
        regime = benchmark[["trade_date", "trend_return_60", "volatility_20"]].copy()
        volatility_median = float(regime["volatility_20"].median())
        regime["trend_regime"] = np.select([regime["trend_return_60"] > 0.05, regime["trend_return_60"] < -0.05], ["bull", "bear"], default="sideways")
        regime["vol_regime"] = np.where(regime["volatility_20"] > volatility_median, "high_vol", "low_vol")
        regime["regime"] = regime["trend_regime"] + "_" + regime["vol_regime"]
        panel = panel.sort_values(["symbol", "trade_date"]).copy()
        panel["future_return"] = panel.groupby("symbol")["close"].shift(-3) / panel["close"] - 1.0
        panel = panel.merge(regime[["trade_date", "regime"]], on="trade_date", how="left")
        factors = raw.get("factor_columns") or ["roc_5", "roc_10", "roc_20", "trend_score_24", "ma_energy_41", "volume_ratio_5_20"]
        factors = [factor for factor in factors if factor in panel.columns]
        rows = []
        for (date, regime_name), group in panel.groupby(["trade_date", "regime"], dropna=True, sort=True):
            for factor in factors:
                future = group[[factor, "future_return"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(future) < 3 or future[factor].nunique() < 2 or future.future_return.nunique() < 2:
                    continue
                ic = future[factor].corr(future.future_return, method="spearman")
                if pd.notna(ic): rows.append({"trade_date": int(date), "regime": str(regime_name), "factor": factor, "rank_ic": float(ic), "n_symbols": int(len(future))})
        daily = pd.DataFrame(rows)
        if daily.empty: raise RuntimeError("没有生成市场阶段因子统计")
        summary = daily.groupby(["regime", "factor"], as_index=False).agg(n_dates=("trade_date", "nunique"), mean_rank_ic=("rank_ic", "mean"), median_rank_ic=("rank_ic", "median"), positive_ic_ratio=("rank_ic", lambda values: float((values > 0).mean())))
        metadata = {"task": "factor_regime", "benchmark": benchmark_symbol, "volatility_median": volatility_median, "factors": factors, "start": start, "end": end, "data_dir": str(data_dir), "hold_days": 3}
        artifacts = []
        for name, kind, content in (("factor_regime_summary.csv", "factor_regime_summary", summary.to_csv(index=False).encode("utf-8-sig")), ("factor_regime_daily.csv", "factor_regime_daily", daily.to_csv(index=False).encode("utf-8-sig")), ("factor_regime_summary.json", "factor_regime_metadata", json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))):
            path, digest, size = _write_artifact(run_id, name, content); artifact_id = record_artifact(run_id, kind, path, digest, size); artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **metadata, "artifacts": artifacts}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
