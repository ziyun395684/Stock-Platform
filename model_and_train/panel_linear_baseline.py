"""Simple leakage-aware linear baseline for panel ranking experiments."""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np

from .panel_training import build_return_targets


class PanelRidgeRanker:
    """Ridge regression baseline using only the last observation in each window.

    The model is deliberately simpler than the LSTM-Transformer. It provides
    a control for whether the panel features and labels contain usable
    cross-sectional signal before adding sequence-model capacity.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.target_mode = "raw"
        self._model = None
        self._target_scale = 0.01

    def fit(self, X: np.ndarray, sample_meta, target_mode: str = "raw") -> "PanelRidgeRanker":
        """Fit only on rows marked ``split == train``."""
        X = np.asarray(X, dtype="float64")
        if X.ndim != 3:
            raise ValueError("X 必须是 [samples, time_steps, features] 三维数组")
        if len(sample_meta) != len(X):
            raise ValueError("sample_meta 行数必须与 X 样本数一致")
        if "split" not in sample_meta.columns:
            raise ValueError("sample_meta 缺少字段: split")
        train_idx = np.flatnonzero(sample_meta["split"].eq("train").to_numpy())
        if len(train_idx) == 0:
            raise ValueError("训练集没有有效样本")
        targets = build_return_targets(sample_meta, train_idx, target_mode)
        self._target_scale = float(np.std(targets))
        if not np.isfinite(self._target_scale) or self._target_scale <= 1e-8:
            self._target_scale = 0.01
        from sklearn.linear_model import Ridge

        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X[train_idx, -1, :], targets)
        self.target_mode = target_mode
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return predictions using the same keys as the multitask model."""
        if self._model is None:
            raise RuntimeError("PanelRidgeRanker: 需先 fit()")
        X = np.asarray(X, dtype="float64")
        if X.ndim != 3:
            raise ValueError("X 必须是 [samples, time_steps, features] 三维数组")
        pred_return = np.asarray(self._model.predict(X[:, -1, :]), dtype="float32")
        event_probability = 1.0 / (
            1.0 + np.exp(-np.clip(pred_return / self._target_scale, -30.0, 30.0))
        )
        return {
            "pred_return": pred_return,
            "event_probability": event_probability.astype("float32"),
        }


def run_panel_ridge_rolling(
    panel,
    windows: Iterable,
    look_back: int,
    universe_symbols: Sequence[str],
    output_dir: str = "output/panel_ridge_rolling",
    top_k: int = 2,
    cost_rate: float = 0.003,
    slippage_rate: float = 0.0,
    initial_capital: float = 1.0,
    hold_days: int = 3,
    label_threshold: float = 0.0,
    target_mode: str = "raw",
    alpha: float = 1.0,
) -> list[dict]:
    """Run the simple Ridge baseline through the shared OOS contract."""
    from data.panel_sequences import build_panel_sequences
    from .panel_rolling import _split_map
    from .panel_inference import aggregate_panel_predictions
    from .panel_evaluation import evaluate_panel_predictions
    from .portfolio_backtest import PortfolioBacktest

    os.makedirs(output_dir, exist_ok=True)
    results = []
    for window in windows:
        seq = build_panel_sequences(
            panel, look_back=look_back, embargo_days=0,
            split_map=_split_map(window), universe_symbols=universe_symbols,
        )
        ranker = PanelRidgeRanker(alpha=alpha).fit(
            seq.X, seq.sample_meta, target_mode=target_mode
        )
        test_idx = seq.test_index
        meta = seq.sample_meta.iloc[test_idx][
            ["trade_date", "symbol", "split", "future_return", "event_label"]
        ].reset_index(drop=True)
        prediction = ranker.predict(seq.X[test_idx])
        prediction_frame = meta.copy()
        prediction_frame["pred_return"] = prediction["pred_return"]
        prediction_frame["event_probability"] = prediction["event_probability"]
        prediction_path = os.path.join(
            output_dir, f"ridge_panel_{int(look_back)}_w{window.index}_predictions.csv"
        )
        prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8-sig")
        signals = aggregate_panel_predictions(
            {look_back: (meta[["trade_date", "symbol"]], prediction)},
            top_k=top_k,
        )
        prices = panel[["trade_date", "symbol", "open", "close"]].copy()
        if "is_suspend" in panel.columns:
            prices["is_suspend"] = panel["is_suspend"]
        report = PortfolioBacktest(
            cost_rate=cost_rate, slippage_rate=slippage_rate,
            initial_capital=initial_capital, hold_days=hold_days,
            label_threshold=label_threshold,
        ).run(prices, signals, end_date=window.test_dates[-1])
        evaluation = evaluate_panel_predictions(prediction_frame)
        all_metrics = evaluation[evaluation["symbol"] == "ALL"]
        metrics = all_metrics.iloc[0].to_dict() if len(all_metrics) else {}
        results.append({
            "window": window.index,
            "target_mode": target_mode,
            "alpha": float(alpha),
            "look_back": int(look_back),
            "train_start": window.train_dates[0],
            "train_end": window.train_dates[-1],
            "val_start": window.val_dates[0],
            "val_end": window.val_dates[-1],
            "test_start": window.test_dates[0],
            "test_end": window.test_dates[-1],
            "n_train_samples": int(len(seq.train_index)),
            "n_val_samples": int(len(seq.val_index)),
            "n_test_samples": int(len(test_idx)),
            "n_trades": report.n_trades,
            "total_return_gross": report.total_return_gross,
            "total_return_net": report.total_return_net,
            "max_drawdown": report.max_drawdown,
            "sharpe": report.sharpe,
            "calmar": report.calmar,
            "hit_rate": report.hit_rate,
            "top_k_hit_rate": report.top_k_hit_rate,
            "turnover": report.turnover,
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "f1": metrics.get("f1"),
            "auc": metrics.get("auc"),
            "rank_ic": metrics.get("rank_ic"),
            "prediction_path": prediction_path,
        })
    return results
