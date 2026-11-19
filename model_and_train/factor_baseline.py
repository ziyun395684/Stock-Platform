"""Interpretable Factor-Only ranking baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factor_library import add_common_factors
from .ranking import rank_and_select


FACTOR_MODES = {"roc", "trend", "ma_energy", "common"}


def _cross_sectional_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True).fillna(0.0)


def build_factor_only_signals(
    panel: pd.DataFrame,
    top_k: int = 2,
    mode: str = "common",
    benchmark_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build daily Top-K signals from common factors only.

    Factor scores are converted to same-day percentile ranks before combining,
    so factors with different numerical scales do not dominate by magnitude.
    """
    if mode not in FACTOR_MODES:
        raise ValueError(f"未知因子模式: {mode}")
    out = add_common_factors(panel, benchmark_frame=benchmark_frame)
    grouped = out.groupby("trade_date", sort=False)

    if mode == "roc":
        out["score"] = grouped["roc_20"].transform(_cross_sectional_rank)
    elif mode == "trend":
        out["score"] = grouped["trend_score_24"].transform(_cross_sectional_rank)
    elif mode == "ma_energy":
        out["score"] = grouped["ma_energy_41"].transform(_cross_sectional_rank)
    else:
        columns = ["roc_5", "roc_20", "trend_score_24", "ma_energy_41", "volume_ratio_5_20"]
        ranks = [grouped[column].transform(_cross_sectional_rank) for column in columns]
        out["score"] = sum(ranks) / len(ranks)

    out["score"] = out["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["factor_mode"] = mode
    return rank_and_select(out, top_k=top_k, score_column="score")
