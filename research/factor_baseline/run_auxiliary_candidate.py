"""Run the Factor-common auxiliary candidate directly from quote CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.auxiliary_signal_cli import run_factor_from_files
from scripts.auxiliary_plan_summary import write_summary


def _read_quote(path: Path, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"open", "close", "volume"}
    missing = required - set(frame.columns)
    if "trade_date" not in frame.columns and "date" not in frame.columns:
        missing.add("date/trade_date")
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    if "trade_date" not in frame.columns:
        frame = frame.rename(columns={"date": "trade_date"})
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    if frame["trade_date"].isna().any():
        raise ValueError(f"{path} 存在无法解析的交易日")
    frame["trade_date"] = frame["trade_date"].dt.strftime("%Y%m%d").astype(int)
    frame["symbol"] = str(symbol)
    if "is_suspend" not in frame.columns:
        frame["is_suspend"] = 0
    return frame[["trade_date", "symbol", "open", "close", "volume", "is_suspend"]]


def _build_panel(data_dir: str | Path, symbols: Sequence[str]) -> pd.DataFrame:
    root = Path(data_dir)
    frames = []
    for symbol in symbols:
        path = root / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(_read_quote(path, symbol))
    if not frames:
        raise ValueError("ETF 池不能为空")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["trade_date", "symbol"]
    ).reset_index(drop=True)


def run_from_quote_directory(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    symbols: Sequence[str],
    benchmark_symbol: str,
    as_of_date,
    top_k: int = 2,
    hold_days: int = 20,
    factor_mode: str = "common",
    cost_rate: float = 0.003,
    initial_capital: float | None = None,
    plan_version: str = "factor-common-v1",
    overwrite: bool = False,
) -> dict:
    normalized_symbols = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    panel = _build_panel(data_dir, normalized_symbols)
    benchmark = _build_panel(data_dir, [benchmark_symbol])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prices_path = output / "prices_panel.csv"
    benchmark_path = output / "benchmark.csv"
    panel.to_csv(prices_path, index=False, encoding="utf-8-sig")
    benchmark[["trade_date", "close"]].to_csv(
        benchmark_path, index=False, encoding="utf-8-sig"
    )
    plan = run_factor_from_files(
        prices_path,
        output,
        as_of_date=as_of_date,
        top_k=top_k,
        hold_days=hold_days,
        factor_mode=factor_mode,
        benchmark_path=benchmark_path,
        cost_rate=cost_rate,
        initial_capital=initial_capital,
        plan_version=plan_version,
        overwrite=overwrite,
        universe_symbols=normalized_symbols,
    )
    write_summary(plan, output / "plan_summary")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="从 quotes CSV 生成 Factor-common 人工复核计划")
    parser.add_argument("--data-dir", default="quotes")
    parser.add_argument("--output-dir", default="output/auxiliary_signal_factor")
    parser.add_argument("--symbols", required=True, help="逗号分隔的 ETF 池")
    parser.add_argument("--benchmark-symbol", default="510300.SH")
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--factor-mode", default="common")
    parser.add_argument("--cost-rate", type=float, default=0.003)
    parser.add_argument("--initial-capital", type=float, default=None)
    parser.add_argument("--plan-version", default="factor-common-v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plan = run_from_quote_directory(
        args.data_dir,
        args.output_dir,
        symbols=args.symbols.split(","),
        benchmark_symbol=args.benchmark_symbol,
        as_of_date=args.as_of_date,
        top_k=args.top_k,
        hold_days=args.hold_days,
        factor_mode=args.factor_mode,
        cost_rate=args.cost_rate,
        initial_capital=args.initial_capital,
        plan_version=args.plan_version,
        overwrite=args.overwrite,
    )
    print({
        "action": plan["action"],
        "signal_date": plan.get("signal_date"),
        "execute_date": plan.get("execute_date"),
        "order_count": len(plan.get("orders", [])),
        "output_dir": args.output_dir,
    })


if __name__ == "__main__":
    main()
