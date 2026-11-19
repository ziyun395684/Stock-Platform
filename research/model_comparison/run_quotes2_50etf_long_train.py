"""Long-train the quotes2 50-ETF Transformer baseline without overwriting the smoke run."""
from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from data.panel import load_universe_frames
from data.panel_pipeline import build_labeled_panel
from model_and_train.config import ModelConfig
from model_and_train.factor_library import add_common_factors
from model_and_train.panel_rolling import run_panel_rolling
from research.model_comparison.run_quotes2_50etf_time_split import make_calendar_window


DATA_DIR = ROOT / "quotes2"
OUTPUT_DIR = ROOT / "output" / "research" / "model_comparison" / "quotes2_50etf_long_train"
BENCHMARK = "510300.SH"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    files = sorted(DATA_DIR.glob("*.csv"))
    symbols = sorted(path.stem for path in files if path.stem != BENCHMARK)
    if len(symbols) != 50:
        raise RuntimeError(f"quotes2 中除基准外应有 50 只 ETF，实际为 {len(symbols)} 只")

    frames = load_universe_frames(
        symbols, data_dir=str(DATA_DIR), start="2013-01-01", end="2026-10-21",
        adjust="none", enforce_min_years=False,
    )
    benchmark = load_universe_frames(
        [BENCHMARK], data_dir=str(DATA_DIR), start="2013-01-01", end="2026-10-21",
        adjust="none", enforce_min_years=False,
    )[BENCHMARK]
    panel = build_labeled_panel(
        frames, benchmark_frame=benchmark, label_threshold=0.0, hold_days=10,
        date_mode="union", extended_features=True,
    )
    panel = add_common_factors(panel, benchmark_frame=benchmark)
    window = make_calendar_window(panel)

    config = ModelConfig(
        lstm_units=64,
        transformer_layers=2,
        num_heads=4,
        transformer_dropout=0.15,
        lstm_dropout=0.2,
        dense_dropout=0.3,
        max_relative_distance=120,
        regression_loss_weight=1.0,
        classification_loss_weight=0.3,
        return_clip=0.2,
        return_target_scaling=True,
        ranking_target="cross_sectional",
        ranking_loss_weight=0.2,
        validation_monitor="rank_ic",
        factor_guidance_weight=0.25,
        factor_guidance_schedule=[0.25, 0.125, 0.0],
        factor_guidance_epochs_per_stage=5,
        indicator_auxiliary_weight=0.1,
        indicator_target_columns=[
            "roc_5", "roc_20", "trend_score_24", "ma_energy_41",
            "volume_ratio_5_20", "relative_return_to_benchmark_20",
        ],
        learning_rate=0.001,
        batch_size=1024,
        epochs=int(args.epochs),
        early_stop_patience=12,
        reduce_lr_patience=6,
        reduce_lr_factor=0.5,
        min_learning_rate=1e-6,
        clipnorm=1.0,
        min_positive_count=50,
        min_positive_ratio=0.04,
        min_sample_count=50,
        random_seed=42,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_panel_rolling(
        panel=panel,
        windows=[window],
        look_back=60,
        universe_symbols=symbols,
        config=config,
        output_dir=str(output_dir / "models"),
        top_k=2,
        cost_rate=0.003,
        slippage_rate=0.0,
        initial_capital=1.0,
        hold_days=10,
        label_threshold=0.0,
        verbose=2,
        include_symbol_one_hot=True,
        calibrate_signal_filter=False,
        feature_columns=None,
        factor_weight=0.0,
        model_weight=1.0,
        exit_mode="fixed",
        experiment_mode="quotes2_50etf_long_train_transformer",
    )
    report = pd.DataFrame(results)
    report.to_csv(output_dir / "test_report.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "data_dir": str(DATA_DIR),
        "benchmark": BENCHMARK,
        "universe_symbols": symbols,
        "universe_size": len(symbols),
        "look_back": 60,
        "hold_days": 10,
        "top_k": 2,
        "cost_rate": 0.003,
        "train_start": window.train_dates[0],
        "train_end": window.train_dates[-1],
        "validation_start": window.val_dates[0],
        "validation_end": window.val_dates[-1],
        "test_start": window.test_dates[0],
        "test_end": window.test_dates[-1],
        "embargo_after_train": list(window.embargo_after_train),
        "embargo_after_val": list(window.embargo_after_val),
        "feature_names": [str(name) for name in panel.columns],
        "model_config": config.to_dict(),
        "training_plan": "10.85",
    }
    metadata["output_dir"] = str(output_dir)
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(report.to_string(index=False))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
