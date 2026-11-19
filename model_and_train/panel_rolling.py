"""Rolling out-of-sample training and portfolio evaluation for ETF panels."""
from __future__ import annotations

import os
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd

from data.panel_sequences import PanelSequenceResult, build_panel_sequences

from .config import ModelConfig
from .multitask_model import MultiTaskLSTMAttentionModel
from .panel_inference import aggregate_panel_predictions
from .panel_training import restore_return_predictions, train_panel_model
from .panel_evaluation import evaluate_panel_predictions
from .portfolio_backtest import PortfolioBacktest
from .ranking import fit_score_weights
from .signal_filter import fit_min_pred_return_threshold
from .rolling import RollingWindow, classify_market_regime
from .prediction_diagnostics import summarize_prediction_dispersion


def _split_map(window: RollingWindow) -> dict:
    mapping = {date: "train" for date in window.train_dates}
    mapping.update({date: "val" for date in window.val_dates})
    mapping.update({date: "test" for date in window.test_dates})
    mapping.update({date: "embargo" for date in window.embargo_after_train})
    mapping.update({date: "embargo" for date in window.embargo_after_val})
    return mapping


def run_panel_rolling(
    panel: pd.DataFrame,
    windows: Iterable[RollingWindow],
    look_back: int,
    universe_symbols: Sequence[str],
    config: Optional[ModelConfig] = None,
    output_dir: str = "model_and_train/checkpoints/rolling",
    top_k: int = 2,
    cost_rate: float = 0.003,
    slippage_rate: float = 0.0,
    initial_capital: float = 1.0,
    hold_days: int = 3,
    label_threshold: float = 0.05,
    train_model_fn: Callable = train_panel_model,
    load_model_fn: Callable = MultiTaskLSTMAttentionModel.load,
    verbose: int = 0,
    include_symbol_one_hot: bool = False,
    calibrate_signal_filter: bool = False,
    signal_filter_candidates: Optional[Sequence[float | None]] = None,
    signal_filter_min_trades: int = 1,
    feature_columns: Optional[Sequence[str]] = None,
    factor_weight: Optional[float] = None,
    model_weight: Optional[float] = None,
    exit_mode: str = "fixed",
    experiment_mode: str = "default",
) -> list[dict]:
    """Train and evaluate each rolling window strictly out of sample."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for window in windows:
        seq: PanelSequenceResult = build_panel_sequences(
            panel,
            look_back=look_back,
            embargo_days=0,
            split_map=_split_map(window),
            feature_columns=feature_columns,
            include_symbol_one_hot=include_symbol_one_hot,
            universe_symbols=universe_symbols,
        )
        path = os.path.join(
            output_dir, f"rolling_panel_{int(look_back)}_w{window.index}.keras"
        )
        train_result = train_model_fn(
            seq,
            universe_symbols=universe_symbols,
            config=config,
            output_path=path,
            label_config={
                "hold_days": int(hold_days),
                "threshold": float(label_threshold),
            },
            verbose=verbose,
        )
        model = load_model_fn(train_result.get("path", path))
        # Test doubles and legacy loaders may not expose model.config. Keep the
        # rolling runner compatible by falling back to the run-level config.
        model_config = getattr(model, "config", None) or config or ModelConfig()
        ranking_target = getattr(model_config, "ranking_target", "raw")
        return_clip = getattr(model_config, "return_clip", 0.2)
        return_target_scaler = getattr(model, "label_config", {}).get("return_target_scaler")
        test_idx = seq.test_index
        meta_columns = ["trade_date", "symbol"]
        if "legacy_factor_score" in seq.sample_meta.columns:
            meta_columns.append("legacy_factor_score")
        meta = seq.sample_meta.iloc[test_idx][meta_columns].reset_index(drop=True)
        ranking_weights = {"pred_return": 1.0, "event_probability": 0.0}
        use_factor_hybrid = factor_weight is not None or model_weight is not None
        if use_factor_hybrid:
            if factor_weight is None or model_weight is None:
                raise ValueError("factor_weight 与 model_weight 必须同时提供")
            ranking_weights = {
                "legacy_factor_score": float(factor_weight),
                "pred_return": float(model_weight),
            }
        signal_filter = None
        signal_filter_threshold = None
        validation_path = os.path.join(
            output_dir, f"rolling_panel_{int(look_back)}_w{window.index}_validation_predictions.csv"
        )
        val_idx = seq.val_index
        if len(val_idx):
            val_columns = [
                "trade_date", "symbol", "split", "future_return", "event_label"
            ]
            if "legacy_factor_score" in seq.sample_meta.columns:
                val_columns.append("legacy_factor_score")
            val_frame = seq.sample_meta.iloc[val_idx][val_columns].reset_index(drop=True)
            val_prediction = restore_return_predictions(
                model.predict(seq.X[val_idx]),
                seq.sample_meta,
                val_idx,
                ranking_target,
                return_clip,
                return_target_scaler,
            )
            val_frame["pred_return"] = val_prediction["pred_return"]
            val_frame["event_probability"] = val_prediction["event_probability"]
            val_frame.to_csv(validation_path, index=False, encoding="utf-8-sig")
            if not use_factor_hybrid:
                try:
                    ranking_weights = fit_score_weights(val_frame)
                except ValueError:
                    ranking_weights = {"pred_return": 1.0, "event_probability": 0.0}
            if calibrate_signal_filter:
                prices_for_validation = panel[[
                    "trade_date", "symbol", "open", "close", "is_suspend"
                ]]
                signal_filter = fit_min_pred_return_threshold(
                    val_frame,
                    prices_for_validation,
                    top_k=top_k,
                    candidates=signal_filter_candidates,
                    cost_rate=cost_rate,
                    slippage_rate=slippage_rate,
                    initial_capital=initial_capital,
                    hold_days=hold_days,
                    label_threshold=label_threshold,
                    min_trades=signal_filter_min_trades,
                )
                signal_filter_threshold = signal_filter["threshold"]
        prediction = restore_return_predictions(
            model.predict(seq.X[test_idx]),
            seq.sample_meta,
            test_idx,
            ranking_target,
            return_clip,
            return_target_scaler,
        )
        signals = aggregate_panel_predictions(
            {look_back: (meta, prediction)},
            top_k=top_k,
            min_pred_return=signal_filter_threshold,
            score_weights=ranking_weights,
        )
        prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]]
        report = PortfolioBacktest(
            cost_rate=cost_rate,
            slippage_rate=slippage_rate,
            initial_capital=initial_capital,
            hold_days=hold_days,
            label_threshold=label_threshold,
            exit_mode=exit_mode,
        ).run(prices, signals, end_date=window.test_dates[-1])
        test_dates = set(window.test_dates)
        regime_frame = panel[panel["trade_date"].isin(test_dates)]
        if {"benchmark_return_1", "volatility_20"}.issubset(regime_frame.columns):
            market_regime = classify_market_regime(
                regime_frame["benchmark_return_1"],
                regime_frame["volatility_20"],
            )
        else:
            market_regime = "unknown"
        prediction_meta_columns = [
            "trade_date", "symbol", "split", "future_return", "event_label"
        ]
        for column in ranking_weights:
            if column in seq.sample_meta.columns and column not in prediction_meta_columns:
                prediction_meta_columns.append(column)
        prediction_frame = seq.sample_meta.iloc[test_idx][
            prediction_meta_columns
        ].reset_index(drop=True)
        prediction_frame["pred_return"] = prediction["pred_return"]
        prediction_frame["event_probability"] = prediction["event_probability"]
        dispersion = summarize_prediction_dispersion(prediction_frame)
        evaluation = evaluate_panel_predictions(prediction_frame)
        all_metrics = evaluation[evaluation["symbol"] == "ALL"]
        metric_row = all_metrics.iloc[0].to_dict() if len(all_metrics) else {}
        prediction_path = os.path.join(
            output_dir, f"rolling_panel_{int(look_back)}_w{window.index}_predictions.csv"
        )
        prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8-sig")
        config_path = os.path.join(
            output_dir, f"rolling_panel_{int(look_back)}_w{window.index}_config.json"
        )
        with open(config_path, "w", encoding="utf-8") as config_file:
            import json
            json.dump({
                "window": window.index,
                "look_back": int(look_back),
                "top_k": int(top_k),
                "cost_rate": float(cost_rate),
                "slippage_rate": float(slippage_rate),
                "initial_capital": float(initial_capital),
                "hold_days": int(hold_days),
                "exit_mode": exit_mode,
                "label_threshold": float(label_threshold),
                "experiment_mode": experiment_mode,
                "indicator_auxiliary_weight": float(
                    getattr(config, "indicator_auxiliary_weight", 0.0)
                ) if config is not None else 0.0,
                "ranking_loss_weight": float(
                    getattr(config, "ranking_loss_weight", 0.0)
                ) if config is not None else 0.0,
                "dispersion_loss_weight": float(
                    getattr(config, "dispersion_loss_weight", 0.0)
                ) if config is not None else 0.0,
                "dispersion_target_std": float(
                    getattr(config, "dispersion_target_std", 0.0)
                ) if config is not None else 0.0,
                "batch_mode": getattr(config, "batch_mode", "auto") if config is not None else "auto",
                "indicator_target_columns": list(
                    getattr(config, "indicator_target_columns", None) or []
                ) if config is not None else [],
                "ranking_weights": ranking_weights,
                "factor_guidance_weights": (
                    list(config.factor_guidance_schedule)
                    if config is not None and config.factor_guidance_schedule is not None
                    else None
                ),
                "signal_filter": signal_filter,
                "train_dates": list(window.train_dates),
                "val_dates": list(window.val_dates),
                "test_dates": list(window.test_dates),
                "embargo_after_train": list(window.embargo_after_train),
                "embargo_after_val": list(window.embargo_after_val),
            }, config_file, ensure_ascii=False, indent=2, default=str)
        results.append({
            "window": window.index,
            "look_back": int(look_back),
            "train_start": window.train_dates[0],
            "train_end": window.train_dates[-1],
            "val_start": window.val_dates[0],
            "val_end": window.val_dates[-1],
            "test_start": window.test_dates[0],
            "test_end": window.test_dates[-1],
            "market_regime": market_regime,
            "ranking_weight_pred_return": ranking_weights.get("pred_return", 0.0),
            "ranking_weight_event_probability": ranking_weights.get("event_probability", 0.0),
            "ranking_weight_legacy_factor": ranking_weights.get("legacy_factor_score", 0.0),
            "experiment_mode": experiment_mode,
            "indicator_auxiliary_weight": float(
                getattr(config, "indicator_auxiliary_weight", 0.0)
            ) if config is not None else 0.0,
            "ranking_loss_weight": float(
                getattr(config, "ranking_loss_weight", 0.0)
            ) if config is not None else 0.0,
            "dispersion_loss_weight": float(
                getattr(config, "dispersion_loss_weight", 0.0)
            ) if config is not None else 0.0,
            "dispersion_target_std": float(
                getattr(config, "dispersion_target_std", 0.0)
            ) if config is not None else 0.0,
            "batch_mode": getattr(config, "batch_mode", "auto") if config is not None else "auto",
            "signal_filter_threshold": signal_filter_threshold,
            "n_train_samples": int(len(seq.train_index)),
            "n_val_samples": int(len(seq.val_index)),
            "n_test_samples": int(len(seq.test_index)),
            "n_trades": report.n_trades,
            "total_return_gross": report.total_return_gross,
            "total_return_net": report.total_return_net,
            "max_drawdown": report.max_drawdown,
            "sharpe": report.sharpe,
            "calmar": report.calmar,
            "hit_rate": report.hit_rate,
            "top_k_hit_rate": report.top_k_hit_rate,
            "turnover": report.turnover,
            "mae": metric_row.get("mae"),
            "rmse": metric_row.get("rmse"),
            "f1": metric_row.get("f1"),
            "auc": metric_row.get("auc"),
            "rank_ic": metric_row.get("rank_ic"),
            "pred_return_std": dispersion["pred_return_std"],
            "event_probability_std": dispersion["event_probability_std"],
            "unique_pred_return_ratio": dispersion["unique_pred_return_ratio"],
            "prediction_collapsed": dispersion["prediction_collapsed"],
            "model_path": train_result.get("path", path),
            "prediction_path": prediction_path,
            "validation_prediction_path": validation_path,
            "config_path": config_path,
        })
    return results
