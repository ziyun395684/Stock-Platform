"""Compare Factor-Only, Composite, Ridge, and Transformer on common OOS windows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_and_train.baseline_adapter import build_composite_baseline_signals
from model_and_train.factor_baseline import FACTOR_MODES, build_factor_only_signals
from model_and_train.pipeline import _load_panel_frames, build_labeled_panel
from model_and_train.portfolio_backtest import PortfolioBacktest


def flatten_config(raw: dict) -> dict:
    return {
        f"{section}.{key}": value
        for section, values in raw.items()
        if isinstance(values, dict)
        for key, value in values.items()
    }


def compare(train_days: int, transformer_path: Path, ridge_path: Path, output_dir: Path) -> pd.DataFrame:
    transformer = pd.read_csv(transformer_path)
    ridge = pd.read_csv(ridge_path)
    key_columns = ["window", "test_start", "test_end"]
    common_keys = transformer[key_columns].drop_duplicates().merge(
        ridge[key_columns].drop_duplicates(), on=key_columns, how="inner"
    )
    if common_keys.empty:
        raise ValueError("Transformer 与 Ridge 没有共同的 OOS 窗口")
    transformer = transformer.merge(common_keys, on=key_columns, how="inner")
    ridge = ridge.merge(common_keys, on=key_columns, how="inner")

    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        cfg = flatten_config(yaml.safe_load(file))
    frames, benchmark = _load_panel_frames(cfg)
    panel = build_labeled_panel(
        frames, benchmark_frame=benchmark, label_threshold=0.0,
        hold_days=10, date_mode="union", extended_features=False,
    )
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]]
    composite = build_composite_baseline_signals(panel, top_k=2)
    factor_signals = {
        mode: build_factor_only_signals(panel, top_k=2, mode=mode, benchmark_frame=benchmark)
        for mode in sorted(FACTOR_MODES)
    }

    rows = []
    for _, key in common_keys.iterrows():
        window = int(key["window"])
        test_start = int(key["test_start"])
        test_end = int(key["test_end"])
        signal_sets = {"Baseline-1 Composite": composite}
        signal_sets.update({f"Factor-{mode}": signals for mode, signals in factor_signals.items()})
        for name, signals in signal_sets.items():
            scoped = signals[(signals["trade_date"] >= test_start) & (signals["trade_date"] <= test_end)]
            report = PortfolioBacktest(cost_rate=0.003, hold_days=10, label_threshold=0.0).run(
                prices, scoped, end_date=test_end
            )
            rows.append({
                "train_days": train_days,
                "window": window,
                "test_start": test_start,
                "test_end": test_end,
                "model": name,
                "total_return_net": report.total_return_net,
                "max_drawdown": report.max_drawdown,
                "n_trades": report.n_trades,
            })
        for name, report_frame in (("Model-2 Ridge", ridge), ("Model-3 Transformer", transformer)):
            record = report_frame[
                (report_frame["window"] == window)
                & (report_frame["test_start"] == test_start)
                & (report_frame["test_end"] == test_end)
            ].iloc[0]
            rows.append({
                "train_days": train_days,
                "window": window,
                "test_start": test_start,
                "test_end": test_end,
                "model": name,
                "total_return_net": record["total_return_net"],
                "max_drawdown": record["max_drawdown"],
                "n_trades": record.get("n_trades", record.get("executed_trades", 0)),
            })
    detail = pd.DataFrame(rows)
    summary = detail.groupby("model", as_index=False).agg(
        mean_net_return=("total_return_net", "mean"),
        positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
        n_windows=("total_return_net", "size"),
        mean_max_drawdown=("max_drawdown", "mean"),
        total_net_return=("total_return_net", lambda values: float((1.0 + values).prod() - 1.0)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "per_window.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-days", type=int, required=True)
    parser.add_argument("--transformer-path", type=Path, required=True)
    parser.add_argument("--ridge-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(args.train_days, args.transformer_path, args.ridge_path, args.output_dir)


if __name__ == "__main__":
    main()
