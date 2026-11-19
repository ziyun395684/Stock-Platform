"""Run a leakage-aware rolling Transformer experiment on factor residual targets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_and_train.panel_rolling import run_panel_rolling
from model_and_train.pipeline import _build_model_config, _load_panel_frames, build_labeled_panel
from model_and_train.rolling import rolling_windows


def flatten_config(raw: dict) -> dict:
    return {
        f"{section}.{key}": value
        for section, values in raw.items()
        if isinstance(values, dict)
        for key, value in values.items()
    }


def run(
    train_days: int = 500,
    val_days: int = 125,
    test_days: int = 125,
    step_days: int = 250,
    max_windows: int | None = None,
    output_dir: str = "output/research/model_comparison/residual_rolling_500/models",
    random_seed: int = 42,
) -> pd.DataFrame:
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
    dates = sorted(panel.loc[panel["valid_sample"].astype(bool), "trade_date"].unique())
    windows = list(rolling_windows(
        dates,
        train_days=train_days,
        val_days=val_days,
        test_days=test_days,
        embargo_days=3,
        step_days=step_days,
    ))
    if max_windows is not None:
        windows = windows[:max_windows]
    local_cfg = dict(cfg)
    local_cfg.update({
        "model.look_backs": [20],
        "model.ranking_target": "factor_residual",
        "model.validation_monitor": "rank_ic",
        "model.return_clip": 0.2,
        "training.epochs": 20,
        "training.random_seed": random_seed,
        "labels.label_threshold": 0.0,
        "backtest.hold_days": 10,
    })
    result = run_panel_rolling(
        panel,
        windows,
        look_back=20,
        universe_symbols=list(cfg["universe.symbols"]),
        config=_build_model_config(local_cfg),
        output_dir=str(ROOT / output_dir),
        top_k=2,
        cost_rate=0.003,
        hold_days=10,
        label_threshold=0.0,
        factor_weight=0.0,
        model_weight=1.0,
        exit_mode="fixed",
    )
    report = pd.DataFrame(result)
    report_dir = ROOT / output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "panel_rolling_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([{
        "model": f"Transformer_factor_residual_{train_days}_seed{random_seed}",
        "train_days": train_days,
        "val_days": val_days,
        "test_days": test_days,
        "step_days": step_days,
        "mean_net_return": report["total_return_net"].mean() if len(report) else None,
        "positive_windows": int((report["total_return_net"] > 0).sum()) if len(report) else 0,
        "n_windows": len(report),
        "mean_max_drawdown": report["max_drawdown"].mean() if len(report) else None,
        "mean_auc": report["auc"].mean() if len(report) else None,
        "mean_rank_ic": report["rank_ic"].mean(skipna=True) if len(report) else None,
        "report_path": str(report_path),
    }])
    summary.to_csv(report_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    if len(report):
        print(report[["window", "test_start", "test_end", "market_regime", "total_return_net", "max_drawdown", "auc", "rank_ic"]].to_string(index=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-days", type=int, default=500)
    parser.add_argument("--val-days", type=int, default=125)
    parser.add_argument("--test-days", type=int, default=125)
    parser.add_argument("--step-days", type=int, default=250)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output-dir", default="output/research/model_comparison/residual_rolling_500/models")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
