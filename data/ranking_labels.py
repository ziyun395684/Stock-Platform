"""Future-return labels for cross-ETF ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .suspend_utils import (
    DEFAULT_HOLD_DAYS,
    ensure_is_suspend,
    hold_window_last_valid_close,
    hold_window_suspension_count,
    next_open_price,
)


def build_ranking_labels(
    frame: pd.DataFrame,
    label_threshold: float = 0.05,
    hold_days: int = DEFAULT_HOLD_DAYS,
    max_suspensions: int = 1,
) -> pd.DataFrame:
    """Attach future return/event labels without changing feature columns."""
    if frame.empty:
        raise ValueError("frame 不能为空")
    required = {"trade_date", "open", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame 缺少字段: {sorted(missing)}")
    df = frame.sort_values("trade_date").reset_index(drop=True).copy()
    susp = ensure_is_suspend(df)
    rows = []
    for pos in range(len(df)):
        buy = next_open_price(df, pos, susp)
        count = hold_window_suspension_count(df, pos, hold_days, susp)
        sell = hold_window_last_valid_close(df, pos, hold_days, susp)
        valid = buy is not None and sell is not None and count <= max_suspensions
        future_return = float(sell / buy - 1.0) if valid else np.nan
        rows.append({
            "buy_price": buy,
            "sell_price": sell,
            "future_return": future_return,
            "event_label": int(valid and future_return >= label_threshold),
            "valid_sample": bool(valid),
            "suspension_count": int(count),
        })
    return pd.concat([df, pd.DataFrame(rows)], axis=1)
