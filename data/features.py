"""因子引擎 / 衍生技术特征 (Task 1.3).

对统一 Schema 日线数据计算衍生特征（仅用当前及过去数据，禁止未来泄露）：

  基础字段：ret(涨跌幅)、aml(涨跌额)
  技术指标：MA5/10/20/60、RSI6/12/24、MACD、BOLL(布林带)、
            换手率(turnover,需流通股本,缺失自动降级)、振幅(amplitude)、
            量比(vol_ratio)、波动率(vol5/vol10)、low_volatility(20日振幅<5%)

设计要点：
  * 所有指标用 rolling(…) 窗口，**不引入未来数据**（时间边界防泄露）。
  * `FactorInterface` 抽象接口 + `register_factor` 注册表：新增外部因子只需实现
    接口并注册，即可被 `compute_factors` 统一调用（对应 Todo「自定义因子接入」）。
  * 依赖缺失（如换手率需要流通股本列）时自动降级：跳过该因子并在 info 中记录，
    对应需求「缺失字段自动降级」。
  * 仅 O(N) 的 rolling/cumsum 运算，无 O(N^2) 循环。
"""
from __future__ import annotations

import abc
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Factor registered table: name -> (required columns, callable(df)->Series)
# ---------------------------------------------------------------------------
_FACTOR_REGISTRY: Dict[str, tuple[tuple[str, ...], Callable[[pd.DataFrame], pd.Series]]] = {}


class FactorInterface(abc.ABC):
    """Base class users extend to plug a custom factor into compute_factors()."""

    name: str = ""

    @abc.abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of factor values indexed like df (no future leakage)."""

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ()


def register_factor(
    name: str,
    required: tuple[str, ...] = (),
):
    """Register a raw function as a factor (decorator)."""

    def deco(func):
        _FACTOR_REGISTRY[name] = (required, func)
        return func

    return deco


# ---------------------------------------------------------------- base fields
@register_factor("ret")
def _ret(df):
    return df["close"].pct_change(fill_method=None)


@register_factor("aml")
def _aml(df):
    return df["close"].diff()


# ---------------------------------------------------------------- MA family
for _w in (5, 10, 20, 60):
    @register_factor(f"ma{_w}", required=("close",))
    def _ma(df, _w=_w):
        return df["close"].rolling(_w).mean()


# ---------------------------------------------------------------- RSI family
def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # when loss==0 the RSI is conventionally 100
    return rsi.where(loss > 0, 100.0).fillna(50.0)


for _w in (6, 12, 24):
    @register_factor(f"rsi{_w}", required=("close",))
    def _rsi_f(df, _w=_w):
        return _rsi(df["close"], _w)


# ---------------------------------------------------------------- MACD
@register_factor("macd", required=("close",))
def _macd(df):
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"macd_dif": dif, "macd_dea": dea, "macd_hist": dif - dea})


# ---------------------------------------------------------------- BOLL
@register_factor("boll", required=("close",))
def _boll(df):
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std(ddof=0)
    return pd.DataFrame(
        {
            "boll_mid": mid,
            "boll_upper": mid + 2 * std,
            "boll_lower": mid - 2 * std,
        }
    )


# ---------------------------------------------------------------- turnover (degrade)
@register_factor("turnover", required=("volume", "float_shares"))
def _turnover(df):
    # 换手率 = 成交量 / 流通股本。float_shares 列缺失时 compute_factors 跳过并降级。
    return df["volume"] / df["float_shares"]


# ---------------------------------------------------------------- amplitude / volume ratio / vol
@register_factor("amplitude", required=("high", "low", "close"))
def _amplitude(df):
    return (df["high"] - df["low"]) / df["close"]


@register_factor("vol_ratio", required=("volume",))
def _vol_ratio(df):
    return df["volume"] / df["volume"].rolling(5).mean()


for _w in (5, 10):
    @register_factor(f"volatility{_w}", required=("close",))
    def _vol(df, _w=_w):
        return df["close"].pct_change(fill_method=None).rolling(_w).std()


# ---------------------------------------------------------------- low volatility (20日振幅<5%)
@register_factor("low_volatility", required=("high", "low", "close"))
def _low_volatility(df):
    hi20 = df["high"].rolling(20).max()
    lo20 = df["low"].rolling(20).min()
    amp20 = (hi20 - lo20) / df["close"]
    return (amp20 < 0.05).astype(int)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_factors(
    df: pd.DataFrame,
    *,
    available_factors: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Compute derived factors on a unified-schema DataFrame.

    Returns (df_with_factors, info). info records which factors were computed
    and which were skipped (degraded) because required columns were missing.

    No future data is used: every factor above is a rolling window ending at
    the current row.
    """
    out = df.copy()
    info: dict = {"computed": [], "degraded": {}, "skipped": []}

    names = [f for f in _FACTOR_REGISTRY]
    if available_factors is not None:
        names = [f for f in names if f in set(available_factors)]

    for name in names:
        required, func = _FACTOR_REGISTRY[name]
        missing = [c for c in required if c not in out.columns]
        if missing:
            info["degraded"][name] = missing
            continue
        try:
            value = _as_frame(func(out), name)
            for col in value.columns:
                out[col] = value[col].to_numpy()
                info["computed"].append(col)
        except Exception as exc:  # noqa: BLE001 - never break the pipeline
            info["skipped"].append(name)
            info["degraded"][name] = [f"error:{type(exc).__name__}"]

    return out, info


def _as_frame(value, name: str):
    if isinstance(value, pd.Series):
        return value.to_frame(name)
    return value


def list_factors() -> list[str]:
    """Names of all registered top-level factors (before column expansion)."""
    return list(_FACTOR_REGISTRY)