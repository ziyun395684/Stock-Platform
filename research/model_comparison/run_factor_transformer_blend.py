"""Evaluate factor/Transformer blend weights on saved rolling predictions."""
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


def run(report_path: Path, output_dir: Path, train_days: int) -> pd.DataFrame:
    report = pd.read_csv(report_path)
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        cfg = flatten_config(yaml.safe_load(file))
    frames, benchmark = _load_panel_frames(cfg)
    panel = build_labeled_panel(
        frames, benchmark_frame=benchmark, label_threshold=0.0,
        hold_days=10, date_mode="union", extended_features=False,
    )
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]]
    factor_signals = build_factor_only_signals(panel, top_k=2, mode="common", benchmark_frame=benchmark)
    factor_scores = factor_signals[["trade_date", "symbol", "score"]].rename(columns={"score": "factor_score"})
    rows = []
    for _, window_row in report.iterrows():
        prediction_path = Path(str(window_row["prediction_path"]))
        if not prediction_path.is_absolute():
            prediction_path = ROOT / prediction_path
        if not prediction_path.exists():
            continue
        predictions = pd.read_csv(prediction_path)
        test_start = int(window_row["test_start"])
        test_end = int(window_row["test_end"])
        predictions = predictions[
            (predictions["trade_date"] >= test_start)
            & (predictions["trade_date"] <= test_end)
        ].copy()
        factors = factor_scores[
            (factor_scores["trade_date"] >= test_start)
            & (factor_scores["trade_date"] <= test_end)
        ].copy()
        for model_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            if model_weight == 0.0:
                signals = rank_and_select(
                    factors.rename(columns={"factor_score": "score"}),
                    top_k=2,
                    score_column="score",
                )
            elif model_weight == 1.0:
                signals = rank_and_select(predictions, top_k=2, score_column="pred_return")
            else:
                blended = blend_model_and_factor_scores(predictions, factors, model_weight=model_weight)
                signals = rank_and_select(blended, top_k=2, score_column="score")
            bt = PortfolioBacktest(cost_rate=0.003, hold_days=10, label_threshold=0.0).run(
                prices, signals, end_date=test_end
            )
            rows.append({
                "train_days": train_days,
                "window": int(window_row["window"]),
                "test_start": test_start,
                "test_end": test_end,
                "model_weight": model_weight,
                "factor_weight": 1.0 - model_weight,
                "total_return_net": bt.total_return_net,
                "max_drawdown": bt.max_drawdown,
                "n_trades": bt.n_trades,
            })
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["model_weight", "factor_weight"], as_index=False).agg(
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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.report, args.output_dir, args.train_days)


if __name__ == "__main__":
    main()
