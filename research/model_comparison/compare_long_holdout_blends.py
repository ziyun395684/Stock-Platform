"""Compare Factor-only and Factor + Transformer blends on long holdout windows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_and_train.factor_baseline import build_factor_only_signals
from model_and_train.hybrid_ranking import blend_model_and_factor_scores
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


def compare(
    report_paths: dict[int, Path],
    output_dir: Path,
    top_k: int = 2,
    cost_rates: tuple[float, ...] = (0.003, 0.005),
    model_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    reports = {seed: pd.read_csv(path) for seed, path in report_paths.items()}
    required = {"window", "test_start", "test_end", "prediction_path"}
    for seed, report in reports.items():
        missing = required - set(report.columns)
        if missing:
            raise ValueError(f"seed={seed} 滚动报告缺少字段: {sorted(missing)}")

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
    factor_signals = build_factor_only_signals(
        panel, top_k=top_k, mode="common", benchmark_frame=benchmark
    )
    factors = factor_signals[["trade_date", "symbol", "score"]].rename(
        columns={"score": "factor_score"}
    )

    rows = []
    for seed, report in reports.items():
        for _, window_row in report.iterrows():
            test_start = int(window_row["test_start"])
            test_end = int(window_row["test_end"])
            window = int(window_row["window"])
            prediction_path = Path(str(window_row["prediction_path"]))
            if not prediction_path.is_absolute():
                prediction_path = ROOT / prediction_path
            predictions = pd.read_csv(prediction_path)
            predictions = predictions[
                (predictions["trade_date"] >= test_start)
                & (predictions["trade_date"] <= test_end)
            ].copy()
            scoped_factors = factors[
                (factors["trade_date"] >= test_start)
                & (factors["trade_date"] <= test_end)
            ].copy()
            for model_weight in model_weights:
                if model_weight == 0.0:
                    signals = rank_and_select(
                        scoped_factors.rename(columns={"factor_score": "score"}),
                        top_k=top_k,
                        score_column="score",
                    )
                elif model_weight == 1.0:
                    signals = rank_and_select(
                        predictions, top_k=top_k, score_column="pred_return"
                    )
                else:
                    blended = blend_model_and_factor_scores(
                        predictions, scoped_factors, model_weight=model_weight
                    )
                    signals = rank_and_select(blended, top_k=top_k, score_column="score")
                for cost_rate in cost_rates:
                    bt = PortfolioBacktest(
                        cost_rate=cost_rate, hold_days=10, label_threshold=0.0
                    ).run(prices, signals, end_date=test_end)
                    rows.append({
                        "seed": seed,
                        "window": window,
                        "test_start": test_start,
                        "test_end": test_end,
                        "model_weight": model_weight,
                        "factor_weight": 1.0 - model_weight,
                        "cost_rate": cost_rate,
                        "total_return_net": bt.total_return_net,
                        "max_drawdown": bt.max_drawdown,
                        "n_trades": bt.n_trades,
                    })

    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["model_weight", "factor_weight", "cost_rate"], as_index=False
    ).agg(
        mean_net_return=("total_return_net", "mean"),
        positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
        n_windows=("total_return_net", "size"),
        mean_max_drawdown=("max_drawdown", "mean"),
        seed_mean_net_return=("total_return_net", "mean"),
    )
    seed_summary = detail.groupby(
        ["seed", "model_weight", "factor_weight", "cost_rate"], as_index=False
    ).agg(
        mean_net_return=("total_return_net", "mean"),
        positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
        n_windows=("total_return_net", "size"),
        mean_max_drawdown=("max_drawdown", "mean"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "per_window.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(output_dir / "by_seed.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-seed42", type=Path, required=True)
    parser.add_argument("--report-seed7", type=Path, required=True)
    parser.add_argument("--report-seed123", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--cost-rates", nargs="+", type=float, default=[0.003, 0.005])
    args = parser.parse_args()
    compare(
        {42: args.report_seed42, 7: args.report_seed7, 123: args.report_seed123},
        args.output_dir,
        top_k=args.top_k,
        cost_rates=tuple(args.cost_rates),
    )


if __name__ == "__main__":
    main()
