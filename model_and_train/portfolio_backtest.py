"""Fixed-horizon equal-weight portfolio backtest for ETF ranking signals."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class PortfolioReport:
    trades: list[dict] = field(default_factory=list)
    nav_curve: list[float] = field(default_factory=list)
    n_trades: int = 0
    total_return_gross: float = 0.0
    total_return_net: float = 0.0
    hit_rate: float | None = None
    max_drawdown: float = 0.0
    turnover: float = 0.0
    signal_count: int = 0
    average_holdings: float = 0.0
    top_k_hit_rate: float | None = None
    annualized_return: float = 0.0
    sharpe: float | None = None
    calmar: float | None = None
    initial_capital: float = 1.0
    final_capital: float = 1.0
    capital_occupied_peak: float = 0.0
    skipped_overlap_signals: int = 0
    slippage_rate: float = 0.0


def _annualized_return(total_return: float, start_date, end_date) -> float:
    """Annualize a return over the actual backtest span."""
    if start_date is None or end_date is None or end_date == start_date:
        return float(total_return)
    try:
        start_text = str(start_date)
        end_text = str(end_date)
        if start_text.isdigit() and end_text.isdigit() and len(start_text) == 8 and len(end_text) == 8:
            elapsed_days = (pd.to_datetime(end_text, format="%Y%m%d") - pd.to_datetime(start_text, format="%Y%m%d")).days
        elif isinstance(start_date, (int, np.integer)) and isinstance(end_date, (int, np.integer)):
            elapsed_days = int(end_date) - int(start_date)
        else:
            elapsed_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    except (TypeError, ValueError):
        elapsed_days = 1
    elapsed_days = max(int(elapsed_days), 1)
    return float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)


class PortfolioBacktest:
    def __init__(
        self,
        cost_rate: float = 0.003,
        hold_days: int = 3,
        label_threshold: float = 0.05,
        exit_mode: str = "fixed",
        stop_loss: float | None = None,
        trend_exit_days: int | None = None,
        rank_exit_threshold: int | None = None,
        max_hold_days: int | None = None,
        slippage_rate: float = 0.0,
        initial_capital: float = 1.0,
    ):
        if cost_rate < 0 or hold_days <= 0:
            raise ValueError("cost_rate 必须非负且 hold_days 必须大于 0")
        if exit_mode not in {"fixed", "dynamic", "signal"}:
            raise ValueError("exit_mode 必须为 fixed、dynamic 或 signal")
        if stop_loss is not None and stop_loss < 0:
            raise ValueError("stop_loss 不能为负数")
        if trend_exit_days is not None and trend_exit_days <= 0:
            raise ValueError("trend_exit_days 必须大于 0")
        if rank_exit_threshold is not None and rank_exit_threshold <= 0:
            raise ValueError("rank_exit_threshold 必须大于 0")
        if max_hold_days is not None and max_hold_days <= 0:
            raise ValueError("max_hold_days 必须大于 0")
        if slippage_rate < 0 or slippage_rate >= 1:
            raise ValueError("slippage_rate 必须在 [0, 1) 范围内")
        if initial_capital <= 0:
            raise ValueError("initial_capital 必须大于 0")
        self.cost_rate = float(cost_rate)
        self.hold_days = int(hold_days)
        self.label_threshold = float(label_threshold)
        self.exit_mode = exit_mode
        self.stop_loss = float(stop_loss) if stop_loss is not None else None
        self.trend_exit_days = int(trend_exit_days) if trend_exit_days is not None else None
        self.rank_exit_threshold = (
            int(rank_exit_threshold) if rank_exit_threshold is not None else None
        )
        self.max_hold_days = int(max_hold_days or hold_days)
        self.slippage_rate = float(slippage_rate)
        self.initial_capital = float(initial_capital)

    def _run_signal_exit(
        self, prices: pd.DataFrame, signals: pd.DataFrame, end_date=None
    ) -> PortfolioReport:
        """Hold selected symbols until they leave Top-K, replacing next open."""
        px = prices.sort_values(["symbol", "trade_date"]).copy()
        px["is_suspend"] = px.get("is_suspend", 0)
        by_symbol = {
            symbol: frame.reset_index(drop=True)
            for symbol, frame in px.groupby("symbol", sort=False)
        }
        evaluation_end = px["trade_date"].max() if end_date is None else end_date
        selected = signals[
            signals["selected"] & (signals["trade_date"] <= evaluation_end)
        ].copy()
        target_by_date = {
            date: set(group["symbol"].astype(str))
            for date, group in selected.groupby("trade_date", sort=True)
        }
        rows: list[dict] = []
        positions: dict[str, dict] = {}
        skipped = 0

        def next_row(symbol: str, date):
            frame = by_symbol.get(symbol)
            if frame is None:
                return None
            candidates = frame[
                (frame["trade_date"] > date)
                & (frame["trade_date"] <= evaluation_end)
            ]
            return candidates.iloc[0] if len(candidates) else None

        def open_position(symbol: str, date) -> None:
            nonlocal skipped
            row = next_row(symbol, date)
            if row is None or int(row["is_suspend"]) != 0 or pd.isna(row["open"]):
                skipped += 1
                return
            price = float(row["open"])
            if price <= 0:
                skipped += 1
                return
            positions[symbol] = {
                "signal_date": date,
                "buy": price * (1.0 + self.slippage_rate),
            }

        def close_position(symbol: str, date, reason: str) -> None:
            position = positions.get(symbol)
            if position is None:
                return
            row = next_row(symbol, date)
            if row is None or int(row["is_suspend"]) != 0 or pd.isna(row["open"]):
                return
            sell = float(row["open"]) * (1.0 - self.slippage_rate)
            if sell <= 0:
                return
            gross = sell / position["buy"] - 1.0
            rows.append({
                "trade_date": position["signal_date"],
                "symbol": symbol,
                "return_gross": gross,
                "return_net": gross - self.cost_rate,
                "hit": gross >= self.label_threshold,
                "exit_date": row["trade_date"],
                "exit_reason": reason,
            })
            del positions[symbol]

        for signal_date, targets in target_by_date.items():
            for symbol in list(positions):
                if symbol not in targets:
                    close_position(symbol, signal_date, "signal_replaced")
            for symbol in sorted(targets):
                if symbol not in positions:
                    open_position(symbol, signal_date)

        for symbol in list(positions):
            frame = by_symbol[symbol]
            final_candidates = frame[frame["trade_date"] <= evaluation_end]
            if final_candidates.empty:
                del positions[symbol]
                continue
            final = final_candidates.iloc[-1]
            position = positions[symbol]
            sell = float(final["close"]) * (1.0 - self.slippage_rate)
            gross = sell / position["buy"] - 1.0
            rows.append({
                "trade_date": position["signal_date"],
                "symbol": symbol,
                "return_gross": gross,
                "return_net": gross - self.cost_rate,
                "hit": gross >= self.label_threshold,
                "exit_date": final["trade_date"],
                "exit_reason": "end_of_backtest",
            })
            del positions[symbol]
        return self._build_report(
            rows,
            len(selected),
            skipped,
            period_start=px["trade_date"].min(),
            period_end=evaluation_end,
        )

    def _build_report(
        self,
        rows: list[dict],
        signal_count: int,
        skipped_overlap_signals: int = 0,
        period_start=None,
        period_end=None,
    ) -> PortfolioReport:
        if not rows:
            return PortfolioReport(
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                signal_count=signal_count,
                skipped_overlap_signals=skipped_overlap_signals,
                slippage_rate=self.slippage_rate,
            )
        trades = pd.DataFrame(rows)
        daily = trades.groupby("trade_date")["return_net"].mean()
        nav = (1.0 + daily).cumprod()
        peak = nav.cummax()
        dd = ((peak - nav) / peak).max()
        daily_gross = trades.groupby("trade_date")["return_gross"].mean()
        gross = (1.0 + daily_gross).prod() - 1.0
        net = (1.0 + daily).prod() - 1.0
        annualized = _annualized_return(net, period_start, period_end)
        daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
        sharpe = float(np.sqrt(252.0) * daily.mean() / daily_std) if daily_std > 0 else None
        calmar = float(annualized / dd) if dd > 0 else None
        return PortfolioReport(
            trades=trades.to_dict("records"), nav_curve=nav.tolist(), n_trades=len(trades),
            total_return_gross=float(gross), total_return_net=float(net),
            hit_rate=float(trades["hit"].mean()), max_drawdown=float(dd),
            turnover=float(len(trades)), signal_count=signal_count,
            annualized_return=annualized, sharpe=sharpe, calmar=calmar,
            initial_capital=self.initial_capital,
            final_capital=float(self.initial_capital * (1.0 + net)),
            capital_occupied_peak=self.initial_capital,
            skipped_overlap_signals=skipped_overlap_signals,
            slippage_rate=self.slippage_rate,
        )

    def run(
        self, prices: pd.DataFrame, signals: pd.DataFrame, end_date=None
    ) -> PortfolioReport:
        required_prices = {"trade_date", "symbol", "open", "close"}
        required_signals = {"trade_date", "symbol", "selected"}
        if not required_prices.issubset(prices.columns):
            raise ValueError(f"prices 缺少字段: {sorted(required_prices - set(prices.columns))}")
        if not required_signals.issubset(signals.columns):
            raise ValueError(f"signals 缺少字段: {sorted(required_signals - set(signals.columns))}")
        if self.exit_mode == "signal":
            return self._run_signal_exit(prices, signals, end_date=end_date)
        px = prices.sort_values(["symbol", "trade_date"]).copy()
        px["is_suspend"] = px.get("is_suspend", 0)
        by_symbol = {s: g.reset_index(drop=True) for s, g in px.groupby("symbol", sort=False)}
        evaluation_end = px["trade_date"].max() if end_date is None else end_date
        rows = []
        held_until: dict[str, object] = {}
        batch_held_until = None
        selected_signals = signals[signals["selected"]].sort_values(
            ["trade_date", "symbol"]
        )
        rank_lookup = {}
        if {"trade_date", "symbol", "rank"}.issubset(signals.columns):
            rank_lookup = {
                (row.trade_date, row.symbol): int(row.rank)
                for row in signals[["trade_date", "symbol", "rank"]].itertuples()
                if pd.notna(row.rank)
            }
        signal_count = int(len(selected_signals))
        batch_returns: list[float] = []
        batch_sizes: list[int] = []
        skipped_overlap_signals = 0
        for signal_date, day_signals in selected_signals.groupby(
            "trade_date", sort=True
        ):
            # First-stage policy: once a Top-K batch is opened, ignore all new
            # signals until every position in that batch has expired.
            if batch_held_until is not None and signal_date <= batch_held_until:
                skipped_overlap_signals += len(day_signals)
                continue
            batch_end = None
            for _, signal in day_signals.iterrows():
                symbol = signal["symbol"]
                if symbol in held_until and signal_date <= held_until[symbol]:
                    skipped_overlap_signals += 1
                    continue
                frame = by_symbol.get(symbol)
                if frame is None:
                    continue
                positions = frame.index[frame["trade_date"] == signal_date].tolist()
                if not positions:
                    continue
                pos = positions[0]
                window = frame.iloc[pos + 1:pos + 1 + self.max_hold_days]
                valid = window[
                    (window["is_suspend"].astype(int) == 0)
                    & (window["close"] > 0)
                ]
                if (
                    len(window) == 0
                    or int(window.iloc[0]["is_suspend"]) != 0
                    or pd.isna(window.iloc[0]["open"])
                    or float(window.iloc[0]["open"]) <= 0
                ):
                    continue
                buy = float(window.iloc[0]["open"]) * (1.0 + self.slippage_rate)
                exit_row = valid.iloc[-1]
                exit_reason = "fixed_horizon" if self.exit_mode == "fixed" else "max_hold"
                down_days = 0
                previous_close = buy
                for _, candidate in valid.iterrows():
                    candidate_return = (
                        float(candidate["close"]) * (1.0 - self.slippage_rate) / buy - 1.0
                    )
                    if self.exit_mode == "dynamic":
                        if self.stop_loss is not None and candidate_return <= -self.stop_loss:
                            exit_row = candidate
                            exit_reason = "stop_loss"
                            break
                        if float(candidate["close"]) < previous_close:
                            down_days += 1
                        else:
                            down_days = 0
                        if (
                            self.trend_exit_days is not None
                            and down_days >= self.trend_exit_days
                        ):
                            exit_row = candidate
                            exit_reason = "trend_exit"
                            break
                        if (
                            self.rank_exit_threshold is not None
                            and rank_lookup.get((candidate["trade_date"], symbol), 0)
                            > self.rank_exit_threshold
                        ):
                            exit_row = candidate
                            exit_reason = "rank_exit"
                            break
                    previous_close = float(candidate["close"])
                sell = float(exit_row["close"]) * (1.0 - self.slippage_rate)
                held_until[symbol] = exit_row["trade_date"]
                batch_end = (
                    exit_row["trade_date"]
                    if batch_end is None
                    else max(batch_end, exit_row["trade_date"])
                )
                gross = sell / buy - 1.0
                rows.append({"trade_date": signal_date, "symbol": symbol,
                             "return_gross": gross, "return_net": gross - self.cost_rate,
                             "hit": gross >= self.label_threshold,
                             "exit_date": exit_row["trade_date"],
                             "exit_reason": exit_reason})
            if batch_end is not None:
                batch_held_until = batch_end
                batch = [row for row in rows if row["trade_date"] == signal_date]
                batch_returns.append(float(np.mean([row["return_gross"] for row in batch])))
                batch_sizes.append(len(batch))
        if not rows:
            return PortfolioReport(
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                skipped_overlap_signals=skipped_overlap_signals,
                slippage_rate=self.slippage_rate,
            )
        trades = pd.DataFrame(rows)
        daily = trades.groupby("trade_date")["return_net"].mean()
        nav = (1.0 + daily).cumprod()
        peak = nav.cummax()
        dd = ((peak - nav) / peak).max()
        daily_gross = trades.groupby("trade_date")["return_gross"].mean()
        gross = (1.0 + daily_gross).prod() - 1.0
        net = (1.0 + daily).prod() - 1.0
        annualized = _annualized_return(
            net,
            px["trade_date"].min(),
            evaluation_end,
        )
        daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
        sharpe = float(np.sqrt(252.0) * daily.mean() / daily_std) if daily_std > 0 else None
        calmar = float(annualized / dd) if dd > 0 else None
        return PortfolioReport(
            trades=trades.to_dict("records"), nav_curve=nav.tolist(), n_trades=len(trades),
            total_return_gross=float(gross), total_return_net=float(net),
            hit_rate=float(trades["hit"].mean()), max_drawdown=float(dd),
            turnover=float(len(trades)),
            signal_count=signal_count,
            average_holdings=float(np.mean(batch_sizes)) if batch_sizes else 0.0,
            top_k_hit_rate=(
                float(np.mean(np.asarray(batch_returns) >= self.label_threshold))
                if batch_returns else None
            ),
            annualized_return=annualized,
            sharpe=sharpe,
            calmar=calmar,
            initial_capital=self.initial_capital,
            final_capital=float(self.initial_capital * (1.0 + net)),
            capital_occupied_peak=(self.initial_capital if rows else 0.0),
            skipped_overlap_signals=skipped_overlap_signals,
            slippage_rate=self.slippage_rate,
        )
