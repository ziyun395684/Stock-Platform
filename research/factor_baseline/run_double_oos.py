"""Run Factor-common on an unseen ETF pool across non-overlapping time windows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.panel import align_etf_frames, load_universe_frames
from model_and_train.factor_baseline import build_factor_only_signals
from model_and_train.portfolio_backtest import PortfolioBacktest


def normalize_trade_dates(values: pd.Series) -> pd.Series:
    """Parse project dates stored as YYYYMMDD integers or date strings."""
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d")
    return pd.to_datetime(series)


def build_non_overlapping_windows(dates: pd.Series, window_size: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return complete, non-overlapping calendar windows from ordered trading dates."""
    if window_size <= 0:
        raise ValueError("window_size 必须大于 0")
    normalized = normalize_trade_dates(dates).dropna().drop_duplicates().sort_values().reset_index(drop=True)
    windows = []
    for start in range(0, len(normalized) - window_size + 1, window_size):
        end = start + window_size - 1
        windows.append((pd.Timestamp(normalized.iloc[start]), pd.Timestamp(normalized.iloc[end])))
    return windows


def run(
    config_path: str = "config.yaml",
    data_dir: str = "quotes",
    symbols: tuple[str, ...] = (),
    benchmark_symbol: str | None = None,
    start: str = "2015-01-01",
    end: str = "2025-11-05",
    window_size: int = 250,
    hold_days: int = 20,
    top_k: int = 2,
    cost_rate: float = 0.005,
) -> pd.DataFrame:
    with Path(config_path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    selected_symbols = list(symbols) or list(config["universe"]["symbols"])
    benchmark_symbol = benchmark_symbol or config["universe"].get("benchmark")
    frames = load_universe_frames(selected_symbols, data_dir=data_dir, start=start, end=end, adjust="none", enforce_min_years=False)
    panel = align_etf_frames(frames, date_mode="union")
    if "is_suspend" not in panel.columns:
        panel["is_suspend"] = 0
    benchmark = None
    if benchmark_symbol:
        benchmark_frames = load_universe_frames([benchmark_symbol], data_dir=data_dir, start=start, end=end, adjust="none", enforce_min_years=False)
        benchmark = align_etf_frames(benchmark_frames, date_mode="union")[["trade_date", "close"]]
    dates = normalize_trade_dates(panel["trade_date"])
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]].copy()
    rows = []
    for window_id, (window_start, window_end) in enumerate(build_non_overlapping_windows(dates, window_size)):
        mask = (dates >= window_start) & (dates <= window_end)
        window_panel = panel.loc[mask].copy()
        window_prices = prices.loc[mask].copy()
        window_benchmark = None if benchmark is None else benchmark[
            (normalize_trade_dates(benchmark["trade_date"]) >= window_start)
            & (normalize_trade_dates(benchmark["trade_date"]) <= window_end)
        ].copy()
        signals = build_factor_only_signals(window_panel, top_k=top_k, mode="common", benchmark_frame=window_benchmark)
        report = PortfolioBacktest(cost_rate=cost_rate, hold_days=hold_days, label_threshold=0.0, exit_mode="fixed").run(
            window_prices,
            signals,
            end_date=int(window_end.strftime("%Y%m%d")),
        )
        rows.append({
            "window_id": window_id,
            "window_start": window_start.date().isoformat(),
            "window_end": window_end.date().isoformat(),
            "universe_size": len(selected_symbols),
            "top_k": top_k,
            "hold_days": hold_days,
            "cost_rate": cost_rate,
            "total_return_net": report.total_return_net,
            "annualized_return": report.annualized_return,
            "max_drawdown": report.max_drawdown,
            "sharpe": report.sharpe,
            "signal_count": report.signal_count,
            "n_trades": report.n_trades,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="未见 ETF 池 + 非重叠时间窗口的双重样本外复测")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="quotes")
    parser.add_argument("--symbols", required=True, help="逗号分隔的外部 ETF")
    parser.add_argument("--benchmark")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-11-05")
    parser.add_argument("--window-size", type=int, default=250)
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--cost-rate", type=float, default=0.005)
    parser.add_argument("--output", default="output/research/factor_baseline/double_oos/report.csv")
    args = parser.parse_args()
    result = run(
        config_path=args.config,
        data_dir=args.data_dir,
        symbols=tuple(item.strip() for item in args.symbols.split(",") if item.strip()),
        benchmark_symbol=args.benchmark,
        start=args.start,
        end=args.end,
        window_size=args.window_size,
        hold_days=args.hold_days,
        top_k=args.top_k,
        cost_rate=args.cost_rate,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
