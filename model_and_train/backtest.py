"""Task 3.4 独立业务回测引擎.

对应 Todo Task3.4 / 需求 §5.3、§8.6、§10.3：

    * `BacktestEngine` / `BacktestReport`（dataclass），复用
      `data.suspend_utils`（停牌判定、T+1 开盘、持仓窗口最后有效收盘）。
    * 规则（与标签生成完全同一口径）：
        - T 日信号 -> T+1 日(P_{t+1}^{开盘})买入；
        - T+1 开盘缺失/停牌/<=0 -> 该信号**丢弃**（不入账）；
        - 持仓窗口(T+1~T+3)恰好停牌 1 日 -> 卖出顺延至最后有效交易日收盘；
        - 区分 gross（毛收益）与 net（净收益=毛收益-摩擦成本）；
        - 摩擦成本可配置（需求§8.6 建议买卖合计 0.3%-0.5%）。
    * **必须与标签生成使用同一清洗、同一复权 DataFrame**（调用方保证，
      与 data.labels 共用 df 才能保证 ret 一致）。
    * 注释：**不模拟涨跌停撮合**（后续迭代）。

指标口径（需求§10.3）：
    * 持仓胜率 Hit Rate = 预测上涨且实际累计收益 ≥ label_threshold
      的成交 / 全部成交；
    * 盈亏比 = 命中成交的平均收益 / 未命中成交的平均亏损（按 gross）；
    * 净值曲线取自 net 收益 cumprod；最大回撤取净值曲线。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from data.suspend_utils import (
    ensure_is_suspend,
    hold_window_last_valid_close,
    next_open_price,
)

from .evaluation import to_json_safe

logger = logging.getLogger(__name__)

#: 默认摩擦成本（买卖合计，需求§8.6 建议 0.3%-0.5%）。
DEFAULT_COST_RATE: float = 0.003
#: 持仓窗口（T+1~T+3）。
DEFAULT_HOLD_DAYS: int = 3
#: 上涨事件阈值（默认 +5%）。
DEFAULT_LABEL_THRESHOLD: float = 0.05


@dataclass
class BacktestReport:
    """回测报告（JSON 安全）。

    Attributes:
        nav_dates: 每笔成交的卖出日（净值曲线时间轴）。
        nav_curve: 净值曲线（net 收益 cumprod，起点 1.0）。
        nav_gross: 毛收益净值曲线（对照）。
        trades: 逐笔成交明细（dict 列表，JSON 安全）。
        n_signals: 信号总数；n_discarded: 因 T+1 开盘缺失/停牌丢弃数。
        n_trades: 实际成交数（= n_signals - n_discarded）。
        total_return_gross / total_return_net: 累计收益。
        hit_rate: 持仓胜率（gross >= label_threshold 的成交占比）。
        win_rate: 简单胜率（net > 0 的成交占比）。
        profit_loss_ratio: 盈亏比（gross）。
        max_drawdown: 净值最大回撤。
        cost_rate: 摩擦成本。
    """

    nav_dates: List[str] = field(default_factory=list)
    nav_curve: List[float] = field(default_factory=list)
    nav_gross: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    n_signals: int = 0
    n_discarded: int = 0
    n_trades: int = 0
    total_return_gross: float = 0.0
    total_return_net: float = 0.0
    hit_rate: Optional[float] = None
    win_rate: Optional[float] = None
    profit_loss_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    cost_rate: float = DEFAULT_COST_RATE
    label_threshold: float = DEFAULT_LABEL_THRESHOLD

    def to_dict(self) -> dict:
        return to_json_safe(self.__dict__)

    def summary(self) -> str:
        return (
            f"BacktestReport(signals={self.n_signals}, trades={self.n_trades}, "
            f"discarded={self.n_discarded}, gross={self.total_return_gross:.4f}, "
            f"net={self.total_return_net:.4f}, hit_rate={self.hit_rate}, "
            f"plr={self.profit_loss_ratio}, max_dd={self.max_drawdown:.4f})"
        )


class BacktestEngine:
    """独立业务回测引擎（Task3.4）。

    Args:
        df: **与标签生成同一份**清洗+复权日线（升序，含 trade_date/open/
            high/low/close/volume，可选 is_suspend）。
        cost_rate: 摩擦成本（买卖合计，默认 0.003）。
        hold_days: 持仓窗口（默认 3）。
        label_threshold: 上涨事件阈值（默认 0.05）。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cost_rate: float = DEFAULT_COST_RATE,
        hold_days: int = DEFAULT_HOLD_DAYS,
        label_threshold: float = DEFAULT_LABEL_THRESHOLD,
    ) -> None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("BacktestEngine: df 必须为非空日线 DataFrame")
        for c in ("trade_date", "open", "close"):
            if c not in df.columns:
                raise ValueError(f"BacktestEngine: df 缺少必要列 '{c}'")
        self.df = df.sort_values("trade_date").reset_index(drop=True)
        self.cost_rate = float(cost_rate)
        self.hold_days = int(hold_days)
        self.label_threshold = float(label_threshold)
        self._is_suspend = ensure_is_suspend(self.df)
        self._date_to_pos = {d: i for i, d in enumerate(self.df["trade_date"].to_numpy())}

    # ------------------------------------------------------------------
    def _execute_signal(self, pos: int) -> Optional[Dict]:
        """执行单笔信号（T 日 -> T+1 买入 -> 持仓窗口最后有效收盘卖出）。

        返回成交 dict 或 None（T+1 开盘缺失/停牌/越界 -> 丢弃）。
        """
        buy = next_open_price(self.df, pos, is_suspend=self._is_suspend)
        if buy is None:
            return None
        sell = hold_window_last_valid_close(
            self.df, pos, days=self.hold_days, is_suspend=self._is_suspend
        )
        if sell is None:
            return None
        gross = sell / buy - 1.0
        net = gross - self.cost_rate
        t_date = self.df["trade_date"].iloc[pos]
        buy_date = self.df["trade_date"].iloc[pos + 1]
        # 卖出日 = 持仓窗口最后一个有效交易日（停牌恰好 1 日顺延）。
        n = len(self.df)
        sell_pos = None
        for k in range(pos + 1, min(pos + 1 + self.hold_days, n)):
            if int(self._is_suspend.iloc[k]) == 0 and self.df["close"].iloc[k] > 0:
                sell_pos = k
        sell_date = (
            self.df["trade_date"].iloc[sell_pos] if sell_pos is not None else t_date
        )
        return {
            "signal_date": str(t_date),
            "buy_date": str(buy_date),
            "sell_date": str(sell_date),
            "buy_price": round(float(buy), 6),
            "sell_price": round(float(sell), 6),
            "return_gross": round(float(gross), 6),
            "return_net": round(float(net), 6),
        }

    def run(
        self,
        signals: pd.DataFrame,
    ) -> BacktestReport:
        """运行回测。

        Args:
            signals: 含 trade_date 与 signal(0/1) 的 DataFrame（信号日期即
                样本日 T，来自标签生成/预测管线）。

        Returns:
            BacktestReport。
        """
        if "trade_date" not in signals.columns or "signal" not in signals.columns:
            raise ValueError("signals 必须含 trade_date 与 signal 列")
        sig = signals.sort_values("trade_date").reset_index(drop=True)
        sig_dates = sig["trade_date"].to_numpy()
        sig_on = np.asarray(sig["signal"].astype(int) == 1)

        trades: List[Dict] = []
        n_discarded = 0
        for d, on in zip(sig_dates, sig_on):
            if not on:
                continue
            pos = self._date_to_pos.get(d)
            if pos is None:
                n_discarded += 1  # 信号日期不在行情内 -> 丢弃
                continue
            t = self._execute_signal(pos)
            if t is None:
                n_discarded += 1
            else:
                trades.append(t)

        n_signals = int(np.sum(sig_on))
        n_trades = len(trades)

        gross_rets = np.array([t["return_gross"] for t in trades]) if trades else np.array([])
        net_rets = np.array([t["return_net"] for t in trades]) if trades else np.array([])

        nav_gross = list(np.cumprod(1.0 + gross_rets)) if len(gross_rets) else []
        nav_net = list(np.cumprod(1.0 + net_rets)) if len(net_rets) else []
        nav_dates = [t["sell_date"] for t in trades]

        total_gross = float(np.prod(1.0 + gross_rets) - 1.0) if len(gross_rets) else 0.0
        total_net = float(np.prod(1.0 + net_rets) - 1.0) if len(net_rets) else 0.0

        # 持仓胜率（gross >= label_threshold）与简单胜率（net > 0）。
        hit_rate = (
            float(np.mean(gross_rets >= self.label_threshold)) if len(gross_rets) else None
        )
        win_rate = float(np.mean(net_rets > 0)) if len(net_rets) else None

        # 盈亏比 = 命中成交平均收益 / 未命中成交平均亏损（gross）。
        plr: Optional[float] = None
        if len(gross_rets):
            wins = gross_rets[gross_rets > 0]
            losses = gross_rets[gross_rets < 0]
            if len(wins) and len(losses):
                plr = float(np.mean(wins) / abs(np.mean(losses)))

        # 最大回撤（net 净值曲线）。
        max_dd = 0.0
        if nav_net:
            peak = -np.inf
            for v in nav_net:
                peak = max(peak, v)
                max_dd = max(max_dd, (peak - v) / peak if peak > 0 else 0.0)

        return BacktestReport(
            nav_dates=nav_dates,
            nav_curve=[round(v, 6) for v in nav_net],
            nav_gross=[round(v, 6) for v in nav_gross],
            trades=trades,
            n_signals=n_signals,
            n_discarded=n_discarded,
            n_trades=n_trades,
            total_return_gross=round(total_gross, 6),
            total_return_net=round(total_net, 6),
            hit_rate=None if hit_rate is None else round(hit_rate, 4),
            win_rate=None if win_rate is None else round(win_rate, 4),
            profit_loss_ratio=None if plr is None else round(plr, 4),
            max_drawdown=round(max_dd, 6),
            cost_rate=self.cost_rate,
            label_threshold=self.label_threshold,
        )

    # ------------------------------------------------------------------
    def actual_return(self, date: object) -> Optional[float]:
        """给定样本日 T 的业务口径实际收益（T+1 开盘买入 -> 窗口最后有效收盘）。

        供"ret_actual 收益比对校验"（TC-BT）使用：应与标签生成器的
        `return` 列完全一致。
        """
        pos = self._date_to_pos.get(date)
        if pos is None:
            return None
        t = self._execute_signal(pos)
        return None if t is None else t["return_gross"]

    def actual_returns(self, dates: Sequence) -> Dict[str, Optional[float]]:
        """批量取实际收益（日期 -> ret_actual）。"""
        return {str(d): self.actual_return(d) for d in dates}
