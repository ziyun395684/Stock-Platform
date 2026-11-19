"""Replay the legacy ``全球资产77_13.py`` strategy with StockRNN components."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.panel import align_etf_frames, load_universe_frames
from model_and_train.legacy_strategy_adapter import build_global_asset_signals
from model_and_train.portfolio_backtest import PortfolioBacktest


SYMBOLS = [
    "162719.SZ", "159915.SZ", "159985.SZ", "515030.SH", "513060.SH",
    "512560.SH", "516510.SH", "515880.SH", "515790.SH", "513100.SH",
    "161128.SZ", "512100.SH", "518880.SH",
]


def replay(
    data_dir: str = "quotes",
    start: str = "2018-01-01",
    end: str = "2025-11-05",
    cost_rate: float = 0.003,
    top_k: int = 1,
    hold_days: int = 10,
) -> pd.DataFrame:
    frames = load_universe_frames(
        SYMBOLS,
        data_dir=data_dir,
        start=start,
        end=end,
        adjust="none",
        enforce_min_years=False,
    )
    panel = align_etf_frames(frames, date_mode="union")
    if "is_suspend" not in panel.columns:
        panel["is_suspend"] = 0
    signals = build_global_asset_signals(panel, top_k=top_k)
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]].copy()

    rows = []
    for name, exit_mode in (("legacy_signal", "signal"), ("fixed_10d", "fixed")):
        report = PortfolioBacktest(
            cost_rate=cost_rate,
            hold_days=hold_days,
            label_threshold=0.0,
            exit_mode=exit_mode,
        ).run(prices, signals, end_date=int(end.replace("-", "")))
        rows.append({
            "strategy": "全球资产77_13",
            "execution_rule": name,
            "start": start,
            "end": end,
            "universe_size": len(SYMBOLS),
            "top_k": top_k,
            "hold_days": hold_days,
            "cost_rate": cost_rate,
            "panel_rows": len(panel),
            "signal_count": report.signal_count,
            "n_trades": report.n_trades,
            "total_return_gross": report.total_return_gross,
            "total_return_net": report.total_return_net,
            "annualized_return": report.annualized_return,
            "max_drawdown": report.max_drawdown,
            "sharpe": report.sharpe,
            "calmar": report.calmar,
            "hit_rate": report.hit_rate,
            "turnover": report.turnover,
            "skipped_overlap_signals": report.skipped_overlap_signals,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="quotes")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-11-05")
    parser.add_argument("--cost-rate", type=float, default=0.003)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument(
        "--output",
        default="output/research/legacy_replay/global_asset77_13_summary.csv",
    )
    args = parser.parse_args()
    replay_args = vars(args).copy()
    replay_args.pop("output")
    result = replay(**replay_args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
