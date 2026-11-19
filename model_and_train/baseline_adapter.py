"""Compatibility adapter for the legacy composite ETF ranking strategy."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .ranking import rank_and_select

def legacy_to_nested_task(legacy: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the old flat Task fields to the current nested Task schema."""
    end_date = legacy.get("end_date") or ""
    return {
        "name": legacy.get("name", "复合排序基线"),
        "task_type": legacy.get("task_type", "etf"),
        "symbols": list(legacy.get("symbols", [])),
        "benchmark": legacy.get("benchmark", "510300.SH"),
        "date": {
            "start_date": legacy.get("start_date", "20100101"),
            "end_date": end_date,
        },
        "factors": {
            "exprs": list(legacy.get("factor_exprs", [])),
            "names": list(legacy.get("factor_names", [])),
        },
        "period": {
            "algo": legacy.get("period", "RunDaily"),
            "period_days": legacy.get("period_days"),
        },
        "selection": {
            "algo": legacy.get("select", "SelectAll"),
            "buy_rules": list(legacy.get("select_buy", [])),
            "buy_at_least_count": int(legacy.get("buy_at_least_count", 1)),
            "sell_rules": list(legacy.get("select_sell", [])),
            "sell_at_least_count": int(legacy.get("sell_at_least_count", 1)),
        },
        "order": {
            "factor": legacy.get("order_by_signal", ""),
            "topK": int(legacy.get("order_by_topK", 1)),
            "dropN": int(legacy.get("order_by_dropN", 0)),
            "is_desc": bool(legacy.get("order_by_DESC", True)),
        },
        "weight": {
            "algo": legacy.get("weight", "WeighEqually"),
            "fixed_weights": dict(legacy.get("weight_fixed", {})),
        },
    }


def composite_ranking_score(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Reproduce the legacy composite ETF ranking formula without future data.

    The legacy expression is
    ``trend_score(close,25)*0.30 + (roc(close,5)+roc(close,10))*0.18
    + ma(volume,5)/ma(volume,20)``.  When the legacy custom factor is absent,
    the causal 25-day return is used as its trend proxy.
    """
    required = {"close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame 缺少字段: {sorted(missing)}")
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    roc5 = close.pct_change(5, fill_method=None)
    roc10 = close.pct_change(10, fill_method=None)
    roc18 = close.pct_change(18, fill_method=None)
    trend = frame["trend_score"] if "trend_score" in frame.columns else close.pct_change(25, fill_method=None)
    volume_ratio = volume.rolling(5, min_periods=1).mean() / volume.rolling(
        20, min_periods=1
    ).mean().replace(0, np.nan)
    score = trend * 0.30 + (roc5 + roc10) * 0.18 + volume_ratio
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # The legacy rule is a sell condition.  With no buy rule, the old engine
    # selects all symbols and removes symbols while the sell rule is true.
    eligible = roc18.fillna(-np.inf) <= 0.15
    return score.astype(float), eligible.astype(bool)


def build_composite_baseline_signals(
    panel: pd.DataFrame,
    top_k: int = 2,
) -> pd.DataFrame:
    """Build the legacy composite strategy's daily rank signal table."""
    required = {"trade_date", "symbol", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")
    from .ranking import rank_and_select

    out = panel.copy()
    out["score"], out["eligible"] = composite_ranking_score(out)
    out = out[out["eligible"]].copy()
    if out.empty:
        out["rank"] = pd.Series(dtype="float64")
        out["selected"] = pd.Series(dtype="bool")
        out["weight"] = pd.Series(dtype="float64")
        return out
    return rank_and_select(out, top_k=top_k, score_column="score")


def _build_factor_baseline_signals(
    panel: pd.DataFrame,
    top_k: int,
    factor: str,
) -> pd.DataFrame:
    required = {"trade_date", "symbol", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")
    out = panel[["trade_date", "symbol", "close"]].copy()
    if factor == "trend":
        out["score"] = out.groupby("symbol")["close"].pct_change(20, fill_method=None)
    elif factor == "momentum":
        ret5 = out.groupby("symbol")["close"].pct_change(5, fill_method=None)
        ret10 = out.groupby("symbol")["close"].pct_change(10, fill_method=None)
        out["score"] = ret5 + ret10
    else:
        raise ValueError(f"未知基线因子: {factor}")
    out["score"] = out["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return rank_and_select(out, top_k=top_k, score_column="score")


def build_trend_baseline_signals(panel: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    return _build_factor_baseline_signals(panel, top_k, "trend")


def build_momentum_baseline_signals(panel: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    return _build_factor_baseline_signals(panel, top_k, "momentum")
