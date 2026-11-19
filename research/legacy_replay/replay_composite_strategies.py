"""Replay the remaining composite legacy ETF strategies."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.panel import align_etf_frames, load_universe_frames
from model_and_train.legacy_strategy_adapter import build_composite_strategy_signals
from model_and_train.portfolio_backtest import PortfolioBacktest


STRATEGIES = {
    "复合排序拟合258_10": {
        "symbols": [
            "518880.SH", "513100.SH", "513500.SH", "162719.SZ", "513330.SH",
            "513520.SH", "588000.SH", "512100.SH", "159531.SZ", "159509.SZ",
            "159813.SZ", "159857.SZ", "159915.SZ", "513730.SH", "159687.SZ",
            "513030.SH", "513180.SH",
        ],
        "start": "2019-01-01",
        "trend_period": 25,
        "sell_roc_window": 18,
        "sell_roc_threshold": 0.15,
        "top_k": 1,
        "use_composite_score": True,
    },
    "复合排序优化_126_15": {
        "symbols": [
            "159915.SZ", "510180.SH", "518880.SH", "513100.SH", "159509.SZ",
            "512100.SH", "513500.SH", "512480.SH", "510050.SH",
        ],
        "start": "2015-01-01",
        "trend_period": 25,
        "sell_roc_window": 21,
        "sell_roc_threshold": 0.17,
        "top_k": 1,
        "use_composite_score": True,
        "trend_weight": 0.4,
        "momentum_weight": 0.2,
    },
    "十年": {
        "symbols": ["513100.SH", "518880.SH", "159915.SZ", "510300.SH"],
        "start": "2013-01-01",
        "trend_period": 24,
        "sell_roc_window": 59,
        "sell_roc_threshold": 0.24,
        "top_k": 1,
        "use_composite_score": False,
    },
}


def replay_one(
    name: str,
    config: dict,
    data_dir: str,
    end: str,
    cost_rate: float,
    hold_days: int,
) -> list[dict]:
    frames = load_universe_frames(
        config["symbols"],
        data_dir=data_dir,
        start=config["start"],
        end=end,
        adjust="none",
        enforce_min_years=False,
    )
    panel = align_etf_frames(frames, date_mode="union")
    if "is_suspend" not in panel.columns:
        panel["is_suspend"] = 0
    signal_config = {key: value for key, value in config.items() if key not in {"symbols", "start"}}
    signals = build_composite_strategy_signals(panel, **signal_config)
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]].copy()
    rows = []
    for execution_rule, exit_mode in (("legacy_signal", "signal"), ("fixed_10d", "fixed")):
        report = PortfolioBacktest(
            cost_rate=cost_rate,
            hold_days=hold_days,
            label_threshold=0.0,
            exit_mode=exit_mode,
        ).run(prices, signals, end_date=int(end.replace("-", "")))
        rows.append({
            "strategy": name,
            "execution_rule": execution_rule,
            "start": config["start"],
            "end": end,
            "universe_size": len(config["symbols"]),
            "top_k": config["top_k"],
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
    return rows


def replay(
    data_dir: str = "quotes",
    end: str = "2025-11-05",
    cost_rate: float = 0.003,
    hold_days: int = 10,
) -> pd.DataFrame:
    rows = []
    for name, config in STRATEGIES.items():
        rows.extend(replay_one(name, config, data_dir, end, cost_rate, hold_days))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="quotes")
    parser.add_argument("--end", default="2025-11-05")
    parser.add_argument("--cost-rate", type=float, default=0.003)
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument(
        "--output",
        default="output/research/legacy_replay/composite_strategies_summary.csv",
    )
    args = parser.parse_args()
    result = replay(
        data_dir=args.data_dir,
        end=args.end,
        cost_rate=args.cost_rate,
        hold_days=args.hold_days,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
