"""Data orchestration for the shared multi-ETF ranking model."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import pandas as pd

from .clean import clean_data
from .panel import align_etf_frames, build_panel_features, load_universe_frames
from .panel_sequences import PanelSequenceResult, build_panel_sequences
from .ranking_labels import build_ranking_labels


def build_labeled_panel(
    frames: Mapping[str, pd.DataFrame],
    benchmark_frame: Optional[pd.DataFrame] = None,
    label_threshold: float = 0.05,
    hold_days: int = 3,
    max_suspensions: int = 1,
    date_mode: str = "common",
    extended_features: bool = False,
) -> pd.DataFrame:
    """Align, feature-engineer, and label a collection of ETF bar frames.

    Labels are generated independently within each symbol so a missing bar in
    one ETF cannot change another ETF's holding window.
    """
    cleaned_frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        cleaned, _ = clean_data(frame, code=str(symbol))
        if not cleaned.empty:
            cleaned_frames[str(symbol)] = cleaned
    if not cleaned_frames:
        raise ValueError("清洗后没有可用的 ETF 数据")

    benchmark = None
    if benchmark_frame is not None:
        benchmark, _ = clean_data(benchmark_frame, code="benchmark")
    aligned = align_etf_frames(cleaned_frames, date_mode=date_mode)
    featured = build_panel_features(
        aligned, benchmark=benchmark, extended_features=extended_features
    )
    labeled = []
    for symbol, group in featured.groupby("symbol", sort=False):
        labeled.append(
            build_ranking_labels(
                group,
                label_threshold=label_threshold,
                hold_days=hold_days,
                max_suspensions=max_suspensions,
            )
        )
    if not labeled:
        raise ValueError("没有可标注的 ETF 数据")
    return pd.concat(labeled, ignore_index=True).sort_values(
        ["trade_date", "symbol"]
    ).reset_index(drop=True)


def prepare_panel_sequences(
    frames: Mapping[str, pd.DataFrame],
    benchmark_frame: Optional[pd.DataFrame] = None,
    look_back: int = 20,
    feature_columns: Optional[Sequence[str]] = None,
    split_ratios: Optional[dict] = None,
    embargo_days: int = 3,
    label_threshold: float = 0.05,
    hold_days: int = 3,
    max_suspensions: int = 1,
    include_symbol_one_hot: bool = False,
    universe_symbols: Optional[Sequence[str]] = None,
    date_mode: str = "common",
    extended_features: bool = False,
) -> PanelSequenceResult:
    """Build leakage-aware shared-model sequences directly from ETF frames."""
    panel = build_labeled_panel(
        frames,
        benchmark_frame=benchmark_frame,
        label_threshold=label_threshold,
        hold_days=hold_days,
        max_suspensions=max_suspensions,
        date_mode=date_mode,
        extended_features=extended_features,
    )
    return build_panel_sequences(
        panel,
        look_back=look_back,
        feature_columns=feature_columns,
        split_ratios=split_ratios,
        embargo_days=embargo_days,
        include_symbol_one_hot=include_symbol_one_hot,
        universe_symbols=universe_symbols,
    )


def load_and_prepare_panel_sequences(
    symbols: Sequence[str],
    data_dir: str = "quotes",
    start: str = "1900-01-01",
    end: str = "2100-01-01",
    adjust: str = "none",
    **kwargs,
) -> PanelSequenceResult:
    """Load configured ETF CSVs and prepare shared-model sequences."""
    frames = load_universe_frames(symbols, data_dir, start, end, adjust)
    return prepare_panel_sequences(frames, **kwargs)
