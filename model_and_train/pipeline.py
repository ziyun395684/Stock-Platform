"""Task 4.3 / 4.4 业务编排（供 main.py CLI 调用）.

把数据层（M1）与模型层（M2/M3）串成 CLI 可驱动的完整业务链路：
    train / predict / backtest / evaluate。

设计约束：
    * 只依赖 `ConfigValidator` 校验后的配置；
    * 数据统一走 `DataSourceFactory`（csv 为默认、无网络依赖）；
    * 清洗/因子/标签/序列/泄露校验全部复用 data 层既有实现；
    * 模型训练/集成/回测/报表复用 model_and_train 各模块；
    * 异常路径输出清晰告警，不静默吞错。
"""
from __future__ import annotations

import logging
import json
import os
import re
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from data.access.factory import DataSourceFactory
from data.clean import clean_data
from data.features import compute_factors
from data.labels import generate_labels
from data.leakage import validate_no_leakage
from data.sequences import build_sequences

from .config_validator import ConfigValidator

logger = logging.getLogger(__name__)


def prepare_panel_sequences(*args, **kwargs):
    """Lazy wrapper to keep legacy CLI imports lightweight."""
    from data.panel_pipeline import prepare_panel_sequences as _prepare

    return _prepare(*args, **kwargs)


def train_panel_model(*args, **kwargs):
    """Lazy wrapper for the shared multi-ETF model trainer."""
    from .panel_training import train_panel_model as _train

    return _train(*args, **kwargs)


def load_panel_model(path: str):
    """Lazy loader for a saved shared panel model."""
    from .multitask_model import MultiTaskLSTMAttentionModel

    return MultiTaskLSTMAttentionModel.load(path)


def aggregate_panel_predictions(*args, **kwargs):
    """Lazy wrapper for multi-window panel prediction aggregation."""
    from .panel_inference import aggregate_panel_predictions as _aggregate

    return _aggregate(*args, **kwargs)


def fit_score_weights(*args, **kwargs):
    """Lazy wrapper for validation-only ranking weight selection."""
    from .ranking import fit_score_weights as _fit

    return _fit(*args, **kwargs)


def build_labeled_panel(*args, **kwargs):
    """Lazy wrapper for aligned ETF features and ranking labels."""
    from data.panel_pipeline import build_labeled_panel as _build

    return _build(*args, **kwargs)


def run_panel_rolling(*args, **kwargs):
    """Lazy wrapper for rolling panel train/evaluate execution."""
    from .panel_rolling import run_panel_rolling as _run

    return _run(*args, **kwargs)


def evaluate_panel_predictions(*args, **kwargs):
    """Lazy wrapper for panel split/symbol evaluation."""
    from .panel_evaluation import evaluate_panel_predictions as _evaluate

    return _evaluate(*args, **kwargs)


def _load_data(cfg: Dict, validator: ConfigValidator) -> pd.DataFrame:
    """按配置加载并清洗日线数据。"""
    data_cfg = {k: v for k, v in cfg.items() if k.startswith("data.")}
    source = data_cfg.get("data.source", "csv")
    code = data_cfg.get("data.code")
    start = data_cfg.get("data.start")
    end = data_cfg.get("data.end")
    adjust_type = data_cfg.get("data.adjust_type", "qfq")
    min_years = data_cfg.get("data.min_years", 5)

    if source == "csv":
        import os

        data_dir = cfg.get("paths.data_dir", "quotes")
        path = os.path.join(data_dir, f"{code}.csv")
        # CSV 适配器用 filepath 构造（DataSourceFactory.create_csv 封装）。
        adapter = DataSourceFactory.create_csv(path)
        df = adapter.fetch(code, start, end, adjust_type)
    else:
        adapter = DataSourceFactory.create(source)
        df = adapter.fetch(code, start, end, adjust_type)

    df_clean, clean_stats = clean_data(df, code=code)
    logger.info("清洗统计: %s", clean_stats)
    # 数据年限校验（需求§5.1：不足 5 年拒绝训练）。
    if len(df_clean) < int(min_years * 250):
        raise RuntimeError(
            f"{code}: 数据不足 {min_years} 年（{len(df_clean)} 行 < {int(min_years * 250)}），拒绝训练"
        )
    return df_clean


def _build_model_config(cfg: Dict) -> Any:
    """从扁平配置构建 ModelConfig（把 model.* / training.* 键映射过去）。"""
    from .config import ModelConfig

    override: Dict = {}
    # 通用 dropout -> 三个 dropout（未单独指定时统一使用）。
    if cfg.get("model.dropout") is not None:
        d = float(cfg["model.dropout"])
        override.setdefault("lstm_dropout", d)
        override.setdefault("transformer_dropout", d)
        override.setdefault("dense_dropout", d)
    for key, attr in [
        ("model.lstm_units", "lstm_units"),
        ("model.transformer_layers", "transformer_layers"),
        ("model.num_heads", "num_heads"),
        ("model.max_relative_distance", "max_relative_distance"),
        ("model.regression_loss_weight", "regression_loss_weight"),
        ("model.classification_loss_weight", "classification_loss_weight"),
        ("model.return_clip", "return_clip"),
        ("model.ranking_target", "ranking_target"),
        ("model.ranking_loss_weight", "ranking_loss_weight"),
        ("model.validation_monitor", "validation_monitor"),
        ("model.factor_guidance_weight", "factor_guidance_weight"),
        ("model.factor_guidance_schedule", "factor_guidance_schedule"),
        ("model.factor_guidance_epochs_per_stage", "factor_guidance_epochs_per_stage"),
        ("training.learning_rate", "learning_rate"),
        ("training.batch_size", "batch_size"),
        ("training.epochs", "epochs"),
        ("training.early_stop_patience", "early_stop_patience"),
        ("training.clipnorm", "clipnorm"),
        ("training.max_class_weight", "max_class_weight"),
        ("training.min_positive_count", "min_positive_count"),
        ("training.min_positive_ratio", "min_positive_ratio"),
        ("training.min_sample_count", "min_sample_count"),
        ("training.parallel_training", "parallel_training"),
        ("training.random_seed", "random_seed"),
    ]:
        if cfg.get(key) is not None:
            override[attr] = cfg[key]
    return ModelConfig.from_dict(override) if override else None


def _load_panel_frames(cfg: Dict) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load configured ETF frames and the benchmark using the CSV adapter."""
    from data.clean import clean_data
    from data.panel import load_universe_frames

    symbols = list(cfg.get("universe.symbols", []))
    benchmark_symbol = cfg.get("universe.benchmark", "510300.SH")
    data_dir = cfg.get("paths.data_dir", "quotes")
    start = cfg.get("data.start", "1900-01-01")
    end = cfg.get("data.end", "2100-01-01")
    adjust = cfg.get("data.adjust_type", "none")
    frames = load_universe_frames(
        symbols,
        data_dir,
        start,
        end,
        adjust,
        enforce_min_years=not bool(cfg.get("universe.allow_short_history", False)),
    )
    benchmark_path = os.path.join(data_dir, f"{benchmark_symbol}.csv")
    benchmark = DataSourceFactory.create_csv(benchmark_path).fetch(
        benchmark_symbol,
        start,
        end,
        adjust,
        enforce_min_years=not bool(cfg.get("universe.allow_short_history", False)),
    )
    benchmark, _ = clean_data(benchmark, code=benchmark_symbol)
    return frames, benchmark


def _save_panel_data_audit(cfg: Dict, frames: Dict[str, pd.DataFrame]) -> str:
    from data.audit import audit_universe_frames

    if all(isinstance(frame, pd.DataFrame) for frame in frames.values()):
        audit = audit_universe_frames(
            frames,
            adjust_type=cfg.get("data.adjust_type", "none"),
        )
    else:
        audit = {
            "status": "skipped_non_dataframe_fixture",
            "symbols": [str(symbol) for symbol in frames],
        }
    path = os.path.join(cfg.get("paths.output_dir", "output"), "panel_data_audit.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2, default=str)
    return path


def _next_panel_model_version(output_dir: str, look_back: int) -> int:
    pattern = re.compile(rf"^multitask_panel_{int(look_back)}_v(\d+)\.keras$")
    versions = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            match = pattern.match(name)
            if match:
                versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def _latest_panel_model_path(output_dir: str, look_back: int) -> str:
    pattern = re.compile(rf"^multitask_panel_{int(look_back)}_v(\d+)\.keras$")
    candidates = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            match = pattern.match(name)
            if match:
                candidates.append((int(match.group(1)), os.path.join(output_dir, name)))
    if not candidates:
        raise RuntimeError(f"look_back={look_back}: 没有可用面板模型")
    return max(candidates, key=lambda item: item[0])[1]


def panel_train_pipeline(cfg: Dict, validator: ConfigValidator, verbose: int = 0) -> Dict:
    """Train one shared multitask model for every configured look-back window."""
    frames, benchmark = _load_panel_frames(cfg)
    audit_path = _save_panel_data_audit(cfg, frames)
    symbols = list(cfg.get("universe.symbols", []))
    output_dir = cfg.get("paths.model_dir", "model_and_train/checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    trained = []
    evaluation_frames = []
    for look_back in cfg.get("model.look_backs", [20, 60, 120]):
        seq = prepare_panel_sequences(
            frames,
            benchmark_frame=benchmark,
            look_back=int(look_back),
            split_ratios=cfg.get("split.split_ratios"),
            embargo_days=cfg.get("split.embargo_days", 3),
            feature_columns=cfg.get("data.panel_feature_columns"),
            extended_features=cfg.get("data.extended_panel_features", False),
            label_threshold=cfg.get("labels.label_threshold", 0.05),
            include_symbol_one_hot=cfg.get("model.use_symbol_one_hot", False),
            universe_symbols=symbols,
            date_mode=cfg.get("universe.date_alignment", "common"),
        )
        version = _next_panel_model_version(output_dir, int(look_back))
        path = os.path.join(
            output_dir, f"multitask_panel_{int(look_back)}_v{version}.keras"
        )
        result = train_panel_model(
            seq,
            universe_symbols=symbols,
            config=_build_model_config(cfg),
            output_path=path,
            label_config={
                "threshold": cfg.get("labels.label_threshold", 0.05),
                "hold_days": cfg.get("backtest.hold_days", 3),
            },
            verbose=verbose,
        )
        if os.path.exists(path):
            model = load_panel_model(path)
            for split in ("train", "val", "test"):
                indices = np.flatnonzero(seq.sample_meta["split"].eq(split).to_numpy())
                if len(indices) == 0:
                    continue
                pred = model.predict(seq.X[indices])
                item = seq.sample_meta.iloc[indices][
                    ["trade_date", "symbol", "split", "future_return", "event_label"]
                ].reset_index(drop=True)
                item["pred_return"] = pred["pred_return"]
                item["event_probability"] = pred["event_probability"]
                evaluation_frames.append(item)
        trained.append({"look_back": int(look_back), "version": version, **result})
    evaluation = None
    evaluation_path = None
    if evaluation_frames:
        evaluation = evaluate_panel_predictions(pd.concat(evaluation_frames, ignore_index=True))
        evaluation_path = os.path.join(
            cfg.get("paths.output_dir", "output"), "panel_train_evaluation.csv"
        )
        os.makedirs(os.path.dirname(os.path.abspath(evaluation_path)), exist_ok=True)
        evaluation.to_csv(evaluation_path, index=False, encoding="utf-8-sig")
    return {
        "status": "trained",
        "universe_symbols": symbols,
        "models": trained,
        "evaluation": evaluation,
        "evaluation_path": evaluation_path,
        "audit_path": audit_path,
    }


def panel_predict_pipeline(cfg: Dict, validator: ConfigValidator) -> Dict:
    """Predict the test segment and emit cross-sectional ETF signals."""
    frames, benchmark = _load_panel_frames(cfg)
    audit_path = _save_panel_data_audit(cfg, frames)
    symbols = list(cfg.get("universe.symbols", []))
    trend_panel = build_labeled_panel(
        frames,
        benchmark_frame=benchmark,
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        date_mode=cfg.get("universe.date_alignment", "common"),
        extended_features=cfg.get("data.extended_panel_features", False),
    )
    lookup_columns = ["trade_date", "symbol", "ret_20"]
    if "legacy_factor_score" in trend_panel.columns:
        lookup_columns.append("legacy_factor_score")
    trend_lookup = trend_panel[lookup_columns].rename(
        columns={"ret_20": "trend_score"}
    )
    if "legacy_factor_score" not in trend_lookup.columns:
        trend_lookup["legacy_factor_score"] = trend_lookup["trend_score"]
    output_dir = cfg.get("paths.model_dir", "model_and_train/checkpoints")
    predictions = {}
    evaluation_frames = []
    validation_frames = []
    for look_back in cfg.get("model.look_backs", [20, 60, 120]):
        seq = prepare_panel_sequences(
            frames,
            benchmark_frame=benchmark,
            look_back=int(look_back),
            split_ratios=cfg.get("split.split_ratios"),
            embargo_days=cfg.get("split.embargo_days", 3),
            feature_columns=cfg.get("data.panel_feature_columns"),
            extended_features=cfg.get("data.extended_panel_features", False),
            label_threshold=cfg.get("labels.label_threshold", 0.05),
            include_symbol_one_hot=cfg.get("model.use_symbol_one_hot", False),
            universe_symbols=symbols,
            date_mode=cfg.get("universe.date_alignment", "common"),
        )
        test_idx = np.asarray(seq.test_index, dtype=int)
        if len(test_idx) == 0:
            continue
        model = load_panel_model(_latest_panel_model_path(output_dir, int(look_back)))
        val_idx = np.asarray(getattr(seq, "val_index", []), dtype=int)
        if len(val_idx) and {"split", "future_return", "event_label"}.issubset(seq.sample_meta.columns):
            val_meta = seq.sample_meta.iloc[val_idx][
                ["trade_date", "symbol", "split", "future_return", "event_label"]
            ].reset_index(drop=True)
            val_meta = val_meta.merge(trend_lookup, on=["trade_date", "symbol"], how="left")
            val_prediction = model.predict(seq.X[val_idx])
            val_meta["pred_return"] = val_prediction["pred_return"]
            val_meta["event_probability"] = val_prediction["event_probability"]
            validation_frames.append(val_meta)
        meta = seq.sample_meta.iloc[test_idx][["trade_date", "symbol"]].reset_index(drop=True)
        meta = meta.merge(trend_lookup, on=["trade_date", "symbol"], how="left")
        prediction = model.predict(seq.X[test_idx])
        predictions[int(look_back)] = (meta, prediction)
        if {"split", "future_return", "event_label"}.issubset(seq.sample_meta.columns):
            evaluation = seq.sample_meta.iloc[test_idx][
                ["trade_date", "symbol", "split", "future_return", "event_label"]
            ].reset_index(drop=True)
            evaluation["pred_return"] = prediction["pred_return"]
            evaluation["event_probability"] = prediction["event_probability"]
            evaluation_frames.append(evaluation)
    ranking_weights = {
        "legacy_factor_score": float(cfg.get("ranking.factor_weight", 0.5)),
        "pred_return": float(cfg.get("ranking.model_weight", 0.5)),
    }
    if validation_frames and cfg.get("ranking.weight_source", "config") == "validation":
        validation_frame = pd.concat(validation_frames, ignore_index=True)
        candidates = [
            {"pred_return": 1.0, "event_probability": 0.0},
            {"pred_return": 0.75, "event_probability": 0.25},
            {"pred_return": 0.5, "event_probability": 0.5},
            {"pred_return": 0.25, "event_probability": 0.75},
            {"pred_return": 0.0, "event_probability": 1.0},
            {"legacy_factor_score": 1.0, "pred_return": 0.0},
            {"legacy_factor_score": 0.75, "pred_return": 0.25},
            {"legacy_factor_score": 0.5, "pred_return": 0.5},
            {"legacy_factor_score": 0.25, "pred_return": 0.75},
            {"pred_return": 0.6, "event_probability": 0.3, "trend_score": 0.1},
            {"pred_return": 0.4, "event_probability": 0.3, "trend_score": 0.3},
        ]
        try:
            ranking_weights = fit_score_weights(validation_frame, candidates=candidates)
        except ValueError as exc:
            logger.warning("验证集评分权重拟合失败，回退配置权重: %s", exc)
    ranking_weights_path = os.path.join(
        cfg.get("paths.output_dir", "output"), "panel_ranking_weights.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(ranking_weights_path)), exist_ok=True)
    with open(ranking_weights_path, "w", encoding="utf-8") as file:
        json.dump({"source": "validation", "weights": ranking_weights}, file, ensure_ascii=False, indent=2)
    signals = aggregate_panel_predictions(
        predictions,
        top_k=int(cfg.get("universe.top_k", 2)),
        min_pred_return=cfg.get("ranking.min_pred_return"),
        min_event_probability=cfg.get("ranking.min_event_probability"),
        score_weights=ranking_weights,
    )
    path = cfg.get("report.output_csv", "output/panel_signals.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    signals.to_csv(path, index=False, encoding="utf-8-sig")
    evaluation = None
    evaluation_path = None
    if evaluation_frames:
        eval_frame = pd.concat(evaluation_frames, ignore_index=True)
        eval_frame = eval_frame.groupby(
            ["trade_date", "symbol", "split", "future_return", "event_label"],
            as_index=False,
        )[["pred_return", "event_probability"]].mean()
        evaluation = evaluate_panel_predictions(eval_frame)
        evaluation_path = os.path.join(
            cfg.get("paths.output_dir", "output"), "panel_evaluation.csv"
        )
        os.makedirs(os.path.dirname(os.path.abspath(evaluation_path)), exist_ok=True)
        evaluation.to_csv(evaluation_path, index=False, encoding="utf-8-sig")
    return {
        "n_samples": len(signals),
        "signals": signals,
        "path": path,
        "evaluation": evaluation,
        "evaluation_path": evaluation_path,
        "audit_path": audit_path,
        "ranking_weights": ranking_weights,
        "ranking_weights_path": ranking_weights_path,
    }


def panel_backtest_pipeline(cfg: Dict, validator: ConfigValidator) -> Dict:
    """Backtest panel prediction signals with the fixed-horizon portfolio engine."""
    from .portfolio_backtest import PortfolioBacktest
    from .portfolio_comparison import compare_top_k
    from .benchmark_baseline import buy_and_hold_report
    from .baseline_adapter import (
        build_composite_baseline_signals,
        build_momentum_baseline_signals,
        build_trend_baseline_signals,
    )
    from .experiment_comparison import build_model_variant_signals, compare_experiments

    prediction = panel_predict_pipeline(cfg, validator)
    frames, benchmark = _load_panel_frames(cfg)
    audit_path = _save_panel_data_audit(cfg, frames)
    panel = build_labeled_panel(
        frames,
        benchmark_frame=benchmark,
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        hold_days=cfg.get("backtest.hold_days", 3),
        date_mode=cfg.get("universe.date_alignment", "common"),
        extended_features=cfg.get("data.extended_panel_features", False),
    )
    report = PortfolioBacktest(
        cost_rate=cfg.get("backtest.cost_rate", 0.003),
        slippage_rate=cfg.get("backtest.slippage_rate", 0.0),
        initial_capital=cfg.get("backtest.initial_capital", 1.0),
        hold_days=cfg.get("backtest.hold_days", 3),
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        exit_mode=cfg.get("backtest.exit_mode", "fixed"),
    ).run(
        panel[["trade_date", "symbol", "open", "close", "volume", "is_suspend"]],
        prediction["signals"],
    )
    output = asdict(report)
    benchmark_report = None
    benchmark_symbol = cfg.get("universe.benchmark", "510300.SH")
    if isinstance(benchmark, pd.DataFrame):
        benchmark_prices = benchmark.copy()
        benchmark_prices["symbol"] = benchmark_symbol
        benchmark_report = buy_and_hold_report(
            benchmark_prices,
            symbol=benchmark_symbol,
            cost_rate=cfg.get("backtest.cost_rate", 0.003),
        )
        output["baseline0_buy_and_hold"] = benchmark_report
    path = cfg.get("report.output_json", "output/panel_report.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2, default=str)
    config_path = os.path.join(
        cfg.get("paths.output_dir", "output"), "panel_backtest_config.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(cfg, file, ensure_ascii=False, indent=2, default=str)
    comparison = compare_top_k(
        panel[["trade_date", "symbol", "open", "close", "volume", "is_suspend"]],
        prediction["signals"],
        top_ks=(1, 2, 3, 5),
        cost_rate=cfg.get("backtest.cost_rate", 0.003),
        slippage_rate=cfg.get("backtest.slippage_rate", 0.0),
        initial_capital=cfg.get("backtest.initial_capital", 1.0),
        hold_days=cfg.get("backtest.hold_days", 3),
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        exit_mode=cfg.get("backtest.exit_mode", "fixed"),
    )
    comparison_path = os.path.join(
        cfg.get("paths.output_dir", "output"), "panel_topk_comparison.csv"
    )
    os.makedirs(os.path.dirname(os.path.abspath(comparison_path)), exist_ok=True)
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    baseline_signals = {
        "Baseline-1": build_composite_baseline_signals(
            panel, top_k=int(cfg.get("universe.top_k", 2))
        ),
        "Baseline-2": build_trend_baseline_signals(
            panel, top_k=int(cfg.get("universe.top_k", 2))
        ),
        "Baseline-3": build_momentum_baseline_signals(
            panel, top_k=int(cfg.get("universe.top_k", 2))
        ),
    }
    model_signals = build_model_variant_signals(
        prediction["signals"],
        top_k=int(cfg.get("universe.top_k", 2)),
        validation_weights=prediction.get("ranking_weights"),
    )
    experiment_table = compare_experiments(
        panel[["trade_date", "symbol", "open", "close", "volume", "is_suspend"]],
        {**baseline_signals, **model_signals},
        cost_rate=cfg.get("backtest.cost_rate", 0.003),
        slippage_rate=cfg.get("backtest.slippage_rate", 0.0),
        initial_capital=cfg.get("backtest.initial_capital", 1.0),
        hold_days=cfg.get("backtest.hold_days", 3),
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        backtest_kwargs={"exit_mode": cfg.get("backtest.exit_mode", "fixed")},
    )
    experiment_path = os.path.join(
        cfg.get("paths.output_dir", "output"), "panel_experiment_comparison.csv"
    )
    experiment_table.to_csv(experiment_path, index=False, encoding="utf-8-sig")
    dynamic_table = None
    dynamic_path = None
    if cfg.get("backtest.dynamic_enabled", False):
        dynamic_table = compare_experiments(
            panel[["trade_date", "symbol", "open", "close", "volume", "is_suspend"]],
            {"Model-3-dynamic": model_signals["Model-3"]},
            cost_rate=cfg.get("backtest.cost_rate", 0.003),
            slippage_rate=cfg.get("backtest.slippage_rate", 0.0),
            initial_capital=cfg.get("backtest.initial_capital", 1.0),
            hold_days=cfg.get("backtest.hold_days", 3),
            label_threshold=cfg.get("labels.label_threshold", 0.05),
            backtest_kwargs={
                "exit_mode": "dynamic",
                "stop_loss": cfg.get("backtest.stop_loss"),
                "trend_exit_days": cfg.get("backtest.trend_exit_days"),
                "rank_exit_threshold": cfg.get("backtest.rank_exit_threshold"),
                "max_hold_days": cfg.get("backtest.max_hold_days", 3),
            },
        )
        dynamic_path = os.path.join(
            cfg.get("paths.output_dir", "output"), "panel_dynamic_exit_comparison.csv"
        )
        dynamic_table.to_csv(dynamic_path, index=False, encoding="utf-8-sig")
    return {
        "report": output,
        "path": path,
        "config_path": config_path,
        "signals": prediction["signals"],
        "comparison": comparison,
        "comparison_path": comparison_path,
        "baseline0": benchmark_report,
        "experiment_comparison": experiment_table,
        "experiment_comparison_path": experiment_path,
        "dynamic_exit_comparison": dynamic_table,
        "dynamic_exit_comparison_path": dynamic_path,
    }


def panel_rolling_pipeline(cfg: Dict, validator: ConfigValidator) -> Dict:
    """Run rolling out-of-sample ETF experiments and save a summary CSV."""
    from .rolling import annotate_regime_report, rolling_windows

    frames, benchmark = _load_panel_frames(cfg)
    audit_path = _save_panel_data_audit(cfg, frames)
    panel = build_labeled_panel(
        frames,
        benchmark_frame=benchmark,
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        hold_days=cfg.get("backtest.hold_days", 3),
        date_mode=cfg.get("universe.date_alignment", "common"),
        extended_features=cfg.get("data.extended_panel_features", False),
    )
    dates = sorted(panel.loc[panel["valid_sample"].astype(bool), "trade_date"].unique())
    windows = list(
        rolling_windows(
            dates,
            train_days=cfg.get("rolling.train_days", 1000),
            val_days=cfg.get("rolling.val_days", 250),
            test_days=cfg.get("rolling.test_days", 250),
            embargo_days=cfg.get("split.embargo_days", 3),
            step_days=cfg.get("rolling.step_days", 250),
        )
    )
    all_results = []
    for look_back in cfg.get("model.look_backs", [20, 60, 120]):
        all_results.extend(
            run_panel_rolling(
                panel,
                windows,
                look_back=int(look_back),
                universe_symbols=list(cfg.get("universe.symbols", [])),
                config=_build_model_config(cfg),
                output_dir=os.path.join(
                    cfg.get("paths.model_dir", "model_and_train/checkpoints"), "rolling"
                ),
                top_k=int(cfg.get("universe.top_k", 2)),
                cost_rate=cfg.get("backtest.cost_rate", 0.003),
                slippage_rate=cfg.get("backtest.slippage_rate", 0.0),
                initial_capital=cfg.get("backtest.initial_capital", 1.0),
                hold_days=cfg.get("backtest.hold_days", 3),
                label_threshold=cfg.get("labels.label_threshold", 0.05),
                include_symbol_one_hot=cfg.get("model.use_symbol_one_hot", False),
                calibrate_signal_filter=cfg.get("rolling.calibrate_signal_filter", False),
                signal_filter_candidates=cfg.get("rolling.signal_filter_candidates"),
                signal_filter_min_trades=cfg.get("rolling.signal_filter_min_trades", 1),
                feature_columns=cfg.get("data.panel_feature_columns"),
                factor_weight=cfg.get("ranking.factor_weight", 0.5),
                model_weight=cfg.get("ranking.model_weight", 0.5),
                exit_mode=cfg.get("backtest.exit_mode", "fixed"),
            )
        )
    result = pd.DataFrame(all_results)
    result = annotate_regime_report(result, panel)
    path = os.path.join(cfg.get("paths.output_dir", "output"), "panel_rolling_report.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "n_windows": len(all_results),
        "results": result,
        "path": path,
        "audit_path": audit_path,
    }


def train_pipeline(cfg: Dict, validator: ConfigValidator, verbose: int = 0) -> Dict:
    """训练全链路：数据 -> 清洗 -> 因子 -> 标签 -> 序列 -> 泄露校验 -> 子模型 -> 集成权重。"""
    from .backtest import BacktestEngine
    from .ensemble import EnsembleModel
    from .report import build_report_json
    from .train_pipeline import train_submodels

    df_clean = _load_data(cfg, validator)
    df_feat, factor_info = compute_factors(df_clean)
    logger.info("因子计算: %s", factor_info)

    labeled = generate_labels(
        df_feat,
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        label_mode=cfg.get("labels.label_mode", "tolerant"),
    )
    res = build_sequences(
        df_feat,
        labeled,
        look_backs=tuple(cfg.get("model.look_backs", [20, 60, 120])),
        split_ratios=cfg.get("split.split_ratios"),
        split_mode=cfg.get("split.split_mode", "base"),
        enable_lowvol_weight=cfg.get("split.enable_lowvol_weight", True),
        embargo_days=cfg.get("split.embargo_days", 3),
    )
    leak = validate_no_leakage(df_feat, labeled, res)
    logger.info("数据泄露校验: %s", "PASS" if leak.passed else "FAIL")
    if not leak.passed:
        raise RuntimeError("数据泄露校验未通过，终止训练（见 LeakageReport）")

    model_dir = cfg.get("paths.model_dir", "model_and_train/checkpoints")
    tr = train_submodels(
        res,
        symbol=cfg.get("data.code", "unknown"),
        config=_build_model_config(cfg),
        output_dir=model_dir,
        parallel_training=cfg.get("training.parallel_training", False),
        seed=cfg.get("training.random_seed"),
        verbose=verbose,
    )

    # 集成权重：baseline 用 val（SequenceResult.val_index 子集预测）。
    ens = EnsembleModel(
        mode=cfg.get("ensemble.mode", "baseline"),
        infer_threshold=cfg.get("ensemble.infer_threshold", 0.5),
    )
    prob_by_lb, y_by_lb, date_by_lb = {}, {}, {}
    for lb in res.look_backs:
        idx = res.val_index[lb]
        if len(idx) == 0:
            continue
        from .lstm_transformer import LSTMAttentionModel

        st = tr["submodel_status"].get(lb, {})
        if st.get("status") != "trained":
            continue
        model = LSTMAttentionModel.load(st["path"])
        prob_by_lb[lb] = model.predict(res.X[lb][idx])
        y_by_lb[lb] = res.y[lb][idx]
        date_by_lb[lb] = res.dates[lb][idx]
    ens.set_window_data(prob_by_lb, date_by_lb, y_by_lb)
    weights = ens.fit_weights()

    report = build_report_json(
        label_mode=cfg.get("labels.label_mode", "tolerant"),
        enable_lowvol_weight=cfg.get("split.enable_lowvol_weight", True),
        embargo_days=cfg.get("split.embargo_days", 3),
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        infer_threshold=cfg.get("ensemble.infer_threshold", 0.5),
        adjust_type=cfg.get("data.adjust_type", "qfq"),
        random_seed=cfg.get("training.random_seed"),
        submodel_status=tr["submodel_status"],
        pos_statistics=tr["pos_statistics"],
        path=cfg.get("report.output_json"),
    )
    return {"weights": weights.to_dict(), "train_result": tr, "report": report}


def predict_pipeline(cfg: Dict, validator: ConfigValidator) -> Dict:
    """预测：加载最新版本子模型 -> 集成 -> 输出 CSV/JSON 报表。"""
    from .ensemble import EnsembleModel
    from .lstm_transformer import LSTMAttentionModel
    from .report import build_report_csv
    from .train_pipeline import load_submodels

    df_clean = _load_data(cfg, validator)
    df_feat, _ = compute_factors(df_clean)
    labeled = generate_labels(
        df_feat,
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        label_mode=cfg.get("labels.label_mode", "tolerant"),
    )
    res = build_sequences(
        df_feat,
        labeled,
        look_backs=tuple(cfg.get("model.look_backs", [20, 60, 120])),
        split_mode=cfg.get("split.split_mode", "base"),
        enable_lowvol_weight=cfg.get("split.enable_lowvol_weight", True),
        embargo_days=cfg.get("split.embargo_days", 3),
    )
    symbol = cfg.get("data.code", "unknown")
    submodels = load_submodels(cfg.get("paths.model_dir"), symbol, res.look_backs)
    if not submodels:
        raise RuntimeError(f"{symbol}: 无可用子模型，无法预测")

    ens = EnsembleModel(
        mode=cfg.get("ensemble.mode", "baseline"),
        infer_threshold=cfg.get("ensemble.infer_threshold", 0.5),
    )
    # 测试集预测（最新 15%，与训练划分一致）。
    prob_by_lb, date_by_lb = {}, {}
    for lb, model in submodels.items():
        idx = res.test_index[lb]
        if len(idx) == 0:
            continue
        prob_by_lb[lb] = model.predict(res.X[lb][idx])
        date_by_lb[lb] = res.dates[lb][idx]
    ens.set_window_data(prob_by_lb, date_by_lb)
    ens_prob, date_dict = ens.predict()
    common = date_dict["aligned"]

    # 对齐 ret_actual / label（按公共日期取）。
    meta = res.sample_meta.set_index("trade_date")
    ret_actual = [float(labeled.df.set_index("trade_date").loc[d, "return"])
                  if d in labeled.df["trade_date"].values else float("nan") for d in common]
    label = [int(meta.loc[d, "label"]) if d in meta.index else 0 for d in common]

    build_report_csv(
        dates=common,
        ret_actual=ret_actual,
        label=label,
        prob_ensemble=ens_prob,
        prob_by_window=date_dict["per_window_aligned_prob"],
        look_backs=res.look_backs,
        path=cfg.get("report.output_csv"),
    )
    return {"n_samples": len(common), "ens_prob": ens_prob.tolist(),
            "date_dict": date_dict}


def backtest_pipeline(cfg: Dict, validator: ConfigValidator) -> Dict:
    """回测：加载子模型 -> 测试集信号 -> BacktestEngine。"""
    from .backtest import BacktestEngine
    from .ensemble import EnsembleModel
    from .train_pipeline import load_submodels

    df_clean = _load_data(cfg, validator)
    df_feat, _ = compute_factors(df_clean)
    labeled = generate_labels(
        df_feat,
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        label_mode=cfg.get("labels.label_mode", "tolerant"),
    )
    res = build_sequences(
        df_feat,
        labeled,
        look_backs=tuple(cfg.get("model.look_backs", [20, 60, 120])),
        split_mode=cfg.get("split.split_mode", "base"),
        enable_lowvol_weight=cfg.get("split.enable_lowvol_weight", True),
    )
    symbol = cfg.get("data.code", "unknown")
    submodels = load_submodels(cfg.get("paths.model_dir"), symbol, res.look_backs)
    if not submodels:
        raise RuntimeError(f"{symbol}: 无可用子模型，无法回测")

    ens = EnsembleModel(
        mode=cfg.get("ensemble.mode", "baseline"),
        infer_threshold=cfg.get("ensemble.infer_threshold", 0.5),
    )
    prob_by_lb, date_by_lb = {}, {}
    for lb, model in submodels.items():
        idx = res.test_index[lb]
        if len(idx) == 0:
            continue
        prob_by_lb[lb] = model.predict(res.X[lb][idx])
        date_by_lb[lb] = res.dates[lb][idx]
    ens.set_window_data(prob_by_lb, date_by_lb)
    ens_prob, date_dict = ens.predict()
    common = date_dict["aligned"]

    signals = pd.DataFrame(
        {"trade_date": common, "signal": ens.predict_binary(ens_prob)}
    )
    engine = BacktestEngine(
        df_clean,
        cost_rate=cfg.get("backtest.cost_rate", 0.003),
        label_threshold=cfg.get("labels.label_threshold", 0.05),
    )
    report = engine.run(signals)
    return {"report": report.to_dict(), "summary": report.summary()}


def evaluate_pipeline(cfg: Dict, validator: ConfigValidator) -> Dict:
    """评估：测试集集成概率 -> 分类指标 + 阈值灵敏度。"""
    from .evaluation import threshold_scan
    from .threshold_sensitivity import threshold_sensitivity_analysis

    pred = predict_pipeline(cfg, validator)
    ens_prob = pred["ens_prob"]
    date_dict = pred["date_dict"]
    common = date_dict["aligned"]

    df_clean = _load_data(cfg, validator)
    df_feat, _ = compute_factors(df_clean)
    labeled = generate_labels(
        df_feat,
        label_threshold=cfg.get("labels.label_threshold", 0.05),
        label_mode=cfg.get("labels.label_mode", "tolerant"),
    )
    meta = labeled.df.set_index("trade_date")
    y_true = [int(meta.loc[d, "label"]) if d in meta.index else 0 for d in common]
    ret = [float(meta.loc[d, "return"]) if d in meta.index else float("nan") for d in common]

    metrics = threshold_scan(y_true, ens_prob)
    sens = threshold_sensitivity_analysis(
        common, y_true, ens_prob, ret,
        min_signal_interval=cfg.get("backtest.min_signal_interval", 0),
        cost_rate=cfg.get("backtest.cost_rate", 0.003),
        label_threshold=cfg.get("labels.label_threshold", 0.05),
    )
    return {"metrics": metrics, "sensitivity": sens}


# ---------------------------------------------------------------------------
def run_command(command: str, cfg_path: str, overrides: Optional[Dict] = None) -> Dict:
    """CLI 命令入口：加载配置 -> 覆盖 -> 校验 -> 分发。"""
    validator = ConfigValidator()
    cfg = validator.load_yaml(cfg_path)
    # 应用命令行覆盖（--set a.b=c）。
    if overrides:
        _apply_overrides(cfg, overrides)
        cfg = validator.validate(cfg)
    handlers = {
        "train": train_pipeline,
        "panel_train": panel_train_pipeline,
        "panel_predict": panel_predict_pipeline,
        "panel_backtest": panel_backtest_pipeline,
        "panel_rolling": panel_rolling_pipeline,
        "predict": predict_pipeline,
        "backtest": backtest_pipeline,
        "evaluate": evaluate_pipeline,
    }
    if command not in handlers:
        raise ValueError(f"未知命令 {command!r}，可用: {sorted(handlers)}")
    return handlers[command](cfg, validator)


def _apply_overrides(cfg: Dict, overrides: Dict[str, Any]) -> None:
    """把 {'a.b': v} 覆盖写入**扁平**配置 dict（ConfigValidator 输出即扁平）。

    注：validate() 返回的点路径扁平结构（如 "data.code"），因此覆盖也按
    点路径键直接写入，避免嵌套结构损坏。
    """
    for dotted, value in overrides.items():
        cfg[dotted] = value
