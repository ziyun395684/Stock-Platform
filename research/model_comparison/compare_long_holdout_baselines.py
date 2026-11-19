"""Compare factor baselines with a saved Transformer on long holdout windows."""
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
from model_and_train.ranking import rank_and_select


def flatten_config(raw: dict) -> dict:
    return {
        f"{section}.{key}": value
        for section, values in raw.items()
        if isinstance(values, dict)
        for key, value in values.items()
    }


def compare(report_path: Path, output_dir: Path, top_k: int = 2, cost_rate: float = 0.003) -> pd.DataFrame:
    report = pd.read_csv(report_path)
    required = {"window", "test_start", "test_end", "prediction_path"}
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"滚动报告缺少字段: {sorted(missing)}")

    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        cfg = flatten_config(yaml.safe_load(file))
    frames, benchmark = _load_panel_frames(cfg)
    panel = build_labeled_panel(
        frames,
        benchmark_frame=benchmark,
        label_threshold=0.0,
        hold_days=10,
        date_mode="union",
        extended_features=False,
    )
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]]
    composite = build_composite_baseline_signals(panel, top_k=top_k)
    factor_signals = {
        mode: build_factor_only_signals(
            panel, top_k=top_k, mode=mode, benchmark_frame=benchmark
        )
        for mode in sorted(FACTOR_MODES)
    }

    rows = []
    for _, window_row in report.iterrows():
        test_start = int(window_row["test_start"])
        test_end = int(window_row["test_end"])
        window = int(window_row["window"])
        signal_sets = {"Baseline-1 Composite": composite}
        signal_sets.update({f"Factor-{mode}": signals for mode, signals in factor_signals.items()})
        for name, signals in signal_sets.items():
            scoped = signals[(signals["trade_date"] >= test_start) & (signals["trade_date"] <= test_end)]
            bt = PortfolioBacktest(cost_rate=cost_rate, hold_days=10, label_threshold=0.0).run(
                prices, scoped, end_date=test_end
            )
            rows.append({
                "window": window,
                "test_start": test_start,
                "test_end": test_end,
                "model": name,
                "total_return_net": bt.total_return_net,
                "max_drawdown": bt.max_drawdown,
                "n_trades": bt.n_trades,
            })

        prediction_path = Path(str(window_row["prediction_path"]))
        if not prediction_path.is_absolute():
            prediction_path = ROOT / prediction_path
        predictions = pd.read_csv(prediction_path)
        predictions = predictions[
            (predictions["trade_date"] >= test_start)
            & (predictions["trade_date"] <= test_end)
        ]
        signals = rank_and_select(predictions, top_k=top_k, score_column="pred_return")
        bt = PortfolioBacktest(cost_rate=cost_rate, hold_days=10, label_threshold=0.0).run(
            prices, signals, end_date=test_end
        )
        rows.append({
            "window": window,
            "test_start": test_start,
            "test_end": test_end,
            "model": "Model-3 Transformer-factor_residual",
            "total_return_net": bt.total_return_net,
            "max_drawdown": bt.max_drawdown,
            "n_trades": bt.n_trades,
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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--cost-rate", type=float, default=0.003)
    args = parser.parse_args()
    compare(args.report, args.output_dir, top_k=args.top_k, cost_rate=args.cost_rate)


if __name__ == "__main__":
    main()
