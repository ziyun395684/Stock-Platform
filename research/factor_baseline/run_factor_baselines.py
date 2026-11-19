"""Run common-factor-only ETF baselines on the current universe."""
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
from model_and_train.factor_baseline import FACTOR_MODES, build_factor_only_signals
from model_and_train.portfolio_backtest import PortfolioBacktest


def run(
    config_path: str = "config.yaml",
    data_dir: str = "quotes",
    start: str = "2015-01-01",
    end: str = "2025-11-05",
    hold_days: int = 10,
    top_ks: tuple[int, ...] = (1, 2),
    cost_rates: tuple[float, ...] = (0.003, 0.005),
) -> pd.DataFrame:
    with Path(config_path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    symbols = list(config["universe"]["symbols"])
    benchmark_symbol = config["universe"].get("benchmark")
    frames = load_universe_frames(
        symbols,
        data_dir=data_dir,
        start=start,
        end=end,
        adjust="none",
        enforce_min_years=False,
    )
    panel = align_etf_frames(frames, date_mode="union")
    if "is_suspend" not in panel.columns:
        panel["is_suspend"] = 0
    benchmark = None
    if benchmark_symbol:
        benchmark_frames = load_universe_frames(
            [benchmark_symbol],
            data_dir=data_dir,
            start=start,
            end=end,
            adjust="none",
            enforce_min_years=False,
        )
        benchmark = align_etf_frames(benchmark_frames, date_mode="union")
        benchmark = benchmark[["trade_date", "close"]]
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]].copy()

    rows = []
    for mode in sorted(FACTOR_MODES):
        for top_k in top_ks:
            signals = build_factor_only_signals(
                panel,
                top_k=top_k,
                mode=mode,
                benchmark_frame=benchmark,
            )
            for cost_rate in cost_rates:
                report = PortfolioBacktest(
                    cost_rate=cost_rate,
                    hold_days=hold_days,
                    label_threshold=0.0,
                    exit_mode="fixed",
                ).run(prices, signals, end_date=int(end.replace("-", "")))
                rows.append({
                    "factor_mode": mode,
                    "top_k": top_k,
                    "start": start,
                    "end": end,
                    "hold_days": hold_days,
                    "cost_rate": cost_rate,
                    "universe_size": len(symbols),
                    "total_return_net": report.total_return_net,
                    "annualized_return": report.annualized_return,
                    "max_drawdown": report.max_drawdown,
                    "sharpe": report.sharpe,
                    "calmar": report.calmar,
                    "hit_rate": report.hit_rate,
                    "turnover": report.turnover,
                    "signal_count": report.signal_count,
                    "n_trades": report.n_trades,
                    "skipped_overlap_signals": report.skipped_overlap_signals,
                })
    return pd.DataFrame(rows)


def _parse_number_list(value: str, converter):
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise ValueError("列表参数不能包含空项")
    return [converter(item) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="quotes")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-11-05")
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--top-ks", default="1,2")
    parser.add_argument("--cost-rates", default="0.003,0.005")
    parser.add_argument("--output", default="output/research/factor_baseline/factor_only_summary.csv")
    args = parser.parse_args()
    result = run(
        args.config,
        args.data_dir,
        args.start,
        args.end,
        args.hold_days,
        tuple(_parse_number_list(args.top_ks, int)),
        tuple(_parse_number_list(args.cost_rates, float)),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
