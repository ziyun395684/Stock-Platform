"""Celery adapter for existing factor Rank IC quality reports."""
from __future__ import annotations

import json
from typing import Any

from .celery_app import celery_app
from .factor_tasks import ROOT, _relative_input, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.factor_quality")
def factor_quality(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the existing factor-effect Rank IC report and register artifacts."""
    _set_run_status(run_id, "running")
    try:
        from data.panel import load_universe_frames
        from data.panel_pipeline import build_labeled_panel
        from model_and_train.factor_library import add_common_factors
        from research.reports.build_quotes2_factor_effect_report import build_report
        from research_platform.backend.app.artifact_repository import record_artifact
        from research_platform.backend.app.multiple_testing import benjamini_hochberg, normal_two_sided_p

        raw = dict(config or {})
        data_dir = _relative_input(raw.get("data_dir", "quotes2"), "quotes2")
        config_path = _relative_input(raw.get("config_path", "config.yaml"), "config.yaml")
        import yaml

        with (ROOT / config_path).open(encoding="utf-8") as handle:
            project_config = yaml.safe_load(handle)
        symbols = list(project_config["universe"]["symbols"])
        benchmark_symbol = str(raw.get("benchmark", project_config["universe"].get("benchmark", "510300.SH")))
        start = str(raw.get("start", "2024-01-01"))
        end = str(raw.get("end", "2024-03-31"))
        hold_days = int(raw.get("hold_days", 3))
        quantiles = int(raw.get("quantiles", 5))
        if quantiles < 2 or hold_days < 1:
            raise ValueError("quantiles 必须至少为 2，hold_days 必须为正数")
        frames = load_universe_frames(symbols, data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)
        benchmark = load_universe_frames([benchmark_symbol], data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)[benchmark_symbol]
        panel = build_labeled_panel(frames, benchmark_frame=benchmark, label_threshold=0.0, hold_days=hold_days, date_mode="union", extended_features=True)
        panel = add_common_factors(panel, benchmark_frame=benchmark)
        dates = sorted(panel["trade_date"].dropna().unique())
        if len(dates) < 3:
            raise ValueError("因子质量评估至少需要 3 个交易日")
        train_end = max(1, int(len(dates) * 0.6))
        val_end = max(train_end + 1, int(len(dates) * 0.8))
        split_by_date = {
            **{date: "train" for date in dates[:train_end]},
            **{date: "val" for date in dates[train_end:val_end]},
            **{date: "test" for date in dates[val_end:]},
        }
        panel["split"] = panel["trade_date"].map(split_by_date)
        summary, yearly, daily = build_report(panel, quantiles=quantiles)
        grouped = daily.groupby(["split", "factor"])["rank_ic"]
        p_values = []
        keys = []
        for key, values in grouped:
            numeric = values.astype(float)
            keys.append(key)
            p_values.append(normal_two_sided_p(float(numeric.mean()), float(numeric.std(ddof=1)), int(numeric.count())))
        q_values = benjamini_hochberg(p_values)
        correction = {key: (p_value, q_value) for key, p_value, q_value in zip(keys, p_values, q_values)}
        summary["p_value"] = [correction.get((row["split"], row["factor"]), (1.0, 1.0))[0] for _, row in summary.iterrows()]
        summary["q_value_fdr"] = [correction.get((row["split"], row["factor"]), (1.0, 1.0))[1] for _, row in summary.iterrows()]
        summary["significant_fdr_05"] = summary["q_value_fdr"] <= 0.05
        artifacts = []
        for name, kind, content in (
            ("factor_quality_summary.csv", "factor_quality_summary", summary.to_csv(index=False).encode("utf-8-sig")),
            ("factor_quality_by_year.csv", "factor_quality_by_year", yearly.to_csv(index=False).encode("utf-8-sig")),
            ("factor_quality_daily.csv", "factor_quality_daily", daily.to_csv(index=False).encode("utf-8-sig")),
        ):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        result = {"task": "factor_quality", "rows": int(len(summary)), "mean_rank_ic": float(summary["mean_rank_ic"].mean()), "multiple_testing": {"method": "Benjamini-Hochberg", "family": "split x factor", "alpha": 0.05, "p_value_method": "normal approximation of daily Rank IC mean"}, "selection_policy": "factor selection must aggregate splits and evidence; never select by one test-window return", "config": {"data_dir": str(data_dir), "start": start, "end": end, "hold_days": hold_days, "quantiles": quantiles}, "artifacts": artifacts}
        path, digest, size = _write_artifact(run_id, "factor_quality_summary.json", json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
        artifact_id = record_artifact(run_id, "factor_quality_metadata", path, digest, size)
        artifacts.append({"id": artifact_id, "artifact_type": "factor_quality_metadata", "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **result}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
