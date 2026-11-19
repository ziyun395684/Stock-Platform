"""Celery adapter for the existing StockRNN LSTM-Transformer rolling trainer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .celery_app import celery_app
from .factor_tasks import ROOT, _write_artifact
from .tasks import _set_run_status


@celery_app.task(bind=True, name="research_platform.transformer_train")
def transformer_train(self, run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the existing time-split Transformer engine and index its outputs."""
    _set_run_status(run_id, "running")
    try:
        import pandas as pd
        from data.panel import load_universe_frames
        from data.panel_pipeline import build_labeled_panel
        from model_and_train.config import ModelConfig
        from model_and_train.factor_library import add_common_factors
        from model_and_train.panel_rolling import run_panel_rolling
        from research.model_comparison.run_quotes2_50etf_time_split import make_calendar_window
        from research_platform.backend.app.artifact_repository import record_artifact
        from research_platform.backend.app.training_metadata import config_digest, validate_training_metadata

        raw = dict(config or {})
        data_dir = Path(raw.get("data_dir", "quotes2"))
        benchmark = str(raw.get("benchmark", "510300.SH"))
        start, end = str(raw.get("start", "2013-01-01")), str(raw.get("end", "2026-10-21"))
        windows = [int(value) for value in raw.get("training_windows", [500, 750, 1000])]
        seeds = [int(value) for value in raw.get("seeds", [7, 42, 123])]
        if any(value not in {500, 750, 1000} for value in windows) or any(value not in {7, 42, 123} for value in seeds):
            raise ValueError("training_windows 只能是 500/750/1000，seeds 只能是 7/42/123")
        symbols = sorted(path.stem for path in (ROOT / data_dir).glob("*.csv") if path.stem != benchmark)
        if not symbols:
            raise ValueError("未找到 Transformer 训练 ETF 数据")
        frames = load_universe_frames(symbols, data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)
        benchmark_frame = load_universe_frames([benchmark], data_dir=str(data_dir), start=start, end=end, adjust="none", enforce_min_years=False)[benchmark]
        panel = add_common_factors(build_labeled_panel(frames, benchmark_frame=benchmark_frame, label_threshold=0.0, hold_days=10, date_mode="union", extended_features=True), benchmark_frame=benchmark)
        model_config = ModelConfig(random_seed=seeds[0])
        output_dir = ROOT / "output" / "platform" / "runs" / run_id / "transformer"
        output_dir.mkdir(parents=True, exist_ok=True)
        results = run_panel_rolling(panel=panel, windows=[make_calendar_window(panel)], look_back=int(raw.get("look_back", 60)), universe_symbols=symbols, config=model_config, output_dir=str(output_dir), top_k=2, cost_rate=float(raw.get("cost_rate", 0.003)), hold_days=10, verbose=1, include_symbol_one_hot=True, experiment_mode="platform_transformer")
        report = pd.DataFrame(results)
        artifacts = []
        for name, content, kind in (("transformer_report.csv", report.to_csv(index=False).encode("utf-8-sig"), "transformer_report"), ("transformer_training_log.json", json.dumps({"status": "succeeded", "windows": windows, "seeds": seeds, "rows": len(report)}, indent=2).encode(), "transformer_training_log")):
            path, digest, size = _write_artifact(run_id, name, content)
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        # Index every engine output: checkpoints, predictions, validation predictions,
        # rolling config, and any transaction/NAV report emitted by the engine.
        for generated_path in output_dir.rglob("*"):
            if not generated_path.is_file():
                continue
            content = generated_path.read_bytes()
            artifact_name = generated_path.relative_to(output_dir).as_posix()
            path, digest, size = _write_artifact(run_id, artifact_name, content)
            suffix = generated_path.suffix.lower()
            kind = "transformer_checkpoint" if suffix in {".keras", ".h5"} else "transformer_engine_output"
            artifact_id = record_artifact(run_id, kind, path, digest, size)
            artifacts.append({"id": artifact_id, "artifact_type": kind, "relative_path": path, "sha256": digest, "size_bytes": size})
        numeric_metrics = {}
        if not report.empty:
            for column in ("total_return_net", "max_drawdown", "rank_ic", "sharpe", "turnover", "n_trades"):
                if column in report.columns:
                    numeric_metrics[f"mean_{column}"] = float(pd.to_numeric(report[column], errors="coerce").mean())
        metadata_config = {"data_dir": str(data_dir), "benchmark": benchmark, "start": start, "end": end, "look_back": int(raw.get("look_back", 60)), "cost_rate": float(raw.get("cost_rate", 0.003))}
        metadata = {"task": "transformer_train", "model_type": "lstm_transformer", "training_windows": windows, "seeds": seeds, "look_back": int(raw.get("look_back", 60)), "metrics": numeric_metrics, "config": metadata_config, "data_version": f"{data_dir}:{start}:{end}", "config_digest": config_digest(metadata_config), "artifact_manifest": artifacts}
        validate_training_metadata(metadata)
        path, digest, size = _write_artifact(run_id, "transformer_training_metadata.json", json.dumps(metadata, indent=2).encode())
        artifact_id = record_artifact(run_id, "transformer_training_metadata", path, digest, size)
        artifacts.append({"id": artifact_id, "artifact_type": "transformer_training_metadata", "relative_path": path, "sha256": digest, "size_bytes": size})
        _set_run_status(run_id, "succeeded")
        return {"status": "succeeded", **metadata}
    except Exception as exc:
        _set_run_status(run_id, "failed", error=exc)
        raise
