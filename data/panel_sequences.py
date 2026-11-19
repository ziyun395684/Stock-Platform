"""Time-series samples for a shared cross-ETF model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


AUXILIARY_INDICATOR_COLUMNS = (
    "roc_5",
    "roc_20",
    "trend_score_24",
    "ma_energy_41",
    "volume_ratio_5_20",
    "relative_return_to_benchmark_20",
)


@dataclass
class PanelSequenceResult:
    X: np.ndarray
    y_return: np.ndarray
    y_event: np.ndarray
    sample_meta: pd.DataFrame
    feature_names: list[str]
    scaler: StandardScaler
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray


def _split_dates(dates: Sequence, train_ratio: float, val_ratio: float, embargo_days: int) -> dict:
    unique = sorted(set(dates))
    n = len(unique)
    train_end = int(round(n * train_ratio))
    val_start = min(n, train_end + max(0, int(embargo_days)))
    val_end = min(n, val_start + int(round(n * val_ratio)))
    test_start = min(n, val_end + max(0, int(embargo_days)))
    result = {}
    for d in unique[:train_end]:
        result[d] = "train"
    for d in unique[train_end:val_start]:
        result[d] = "embargo"
    for d in unique[val_start:val_end]:
        result[d] = "val"
    for d in unique[val_end:test_start]:
        result[d] = "embargo"
    for d in unique[test_start:]:
        result[d] = "test"
    return result


def build_panel_sequences(
    panel: pd.DataFrame,
    look_back: int = 20,
    feature_columns: Optional[Sequence[str]] = None,
    split_ratios: Optional[dict] = None,
    embargo_days: int = 3,
    split_map: Optional[Mapping] = None,
    include_symbol_one_hot: bool = False,
    universe_symbols: Optional[Sequence[str]] = None,
) -> PanelSequenceResult:
    """Build shared-model windows with date-level train/val/test isolation."""
    if look_back <= 0:
        raise ValueError("look_back 必须大于 0")
    required = {"trade_date", "symbol", "future_return", "event_label", "valid_sample"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段: {sorted(missing)}")
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    if split_ratios:
        ratios.update(split_ratios)
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("split_ratios 之和必须为 1")
    frame = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    if include_symbol_one_hot:
        symbols = list(universe_symbols or sorted(frame["symbol"].astype(str).unique()))
        if not symbols:
            raise ValueError("启用 symbol one-hot 时 universe_symbols 不能为空")
        for symbol in symbols:
            frame[f"symbol_one_hot_{symbol}"] = (
                frame["symbol"].astype(str) == str(symbol)
            ).astype(float)
    if feature_columns is None:
        excluded = {
            "trade_date",
            "symbol",
            "future_return",
            "event_label",
            "valid_sample",
            "buy_price",
            "sell_price",
            "suspension_count",
            "is_suspend",
        }
        feature_columns = [
            c for c in frame.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
        ]
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("没有可用的数值特征")
    if split_map is None:
        split_map = _split_dates(
            frame.loc[frame["valid_sample"].astype(bool), "trade_date"].tolist(),
            ratios["train"], ratios["val"], embargo_days,
        )
    else:
        split_map = dict(split_map)
    samples = []
    meta = []
    for symbol, group in frame.groupby("symbol", sort=False):
        group = group.reset_index(drop=True)
        values = group[feature_columns].ffill().fillna(0.0).to_numpy(dtype="float64")
        for pos in range(look_back - 1, len(group)):
            row = group.iloc[pos]
            if not bool(row["valid_sample"]):
                continue
            split = split_map.get(row["trade_date"], "embargo")
            samples.append(values[pos - look_back + 1:pos + 1])
            sample_info = {
                "symbol": symbol,
                "trade_date": row["trade_date"],
                "future_return": float(row["future_return"]),
                "event_label": int(row["event_label"]),
                "split": split,
            }
            if "legacy_factor_score" in group.columns:
                sample_info["legacy_factor_score"] = float(
                    pd.to_numeric(row["legacy_factor_score"], errors="coerce")
                )
            for column in AUXILIARY_INDICATOR_COLUMNS:
                if column in group.columns:
                    sample_info[column] = float(
                        pd.to_numeric(row[column], errors="coerce")
                    )
            meta.append(sample_info)
    if not samples:
        raise ValueError("没有足够的有效面板序列样本")
    X = np.asarray(samples, dtype="float64")
    sample_meta = pd.DataFrame(meta)
    train_mask = sample_meta["split"].eq("train").to_numpy()
    if not train_mask.any():
        raise ValueError("训练集没有有效面板序列样本")
    scaler = StandardScaler()
    scaler.fit(X[train_mask].reshape(-1, X.shape[-1]))
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype("float32")
    train_index = np.flatnonzero(sample_meta["split"].eq("train").to_numpy())
    val_index = np.flatnonzero(sample_meta["split"].eq("val").to_numpy())
    test_index = np.flatnonzero(sample_meta["split"].eq("test").to_numpy())
    return PanelSequenceResult(
        X=X,
        y_return=sample_meta["future_return"].to_numpy(dtype="float32"),
        y_event=sample_meta["event_label"].to_numpy(dtype="float32"),
        sample_meta=sample_meta,
        feature_names=feature_columns,
        scaler=scaler,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
    )
