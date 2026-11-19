"""Run a common-window comparison of indicator-learning experiment arms.

The command is intentionally explicit about the four arms:
``raw_only``, ``explicit_indicators``, ``indicator_auxiliary`` and
``factor_only``.  Model arms use :func:`run_panel_rolling`; the factor arm is
replayed with the same portfolio evaluator on the same test windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def ensure_project_root() -> None:
    """Make direct file execution resolve imports from the project root."""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

ensure_project_root()
from model_and_train.config import ModelConfig
from model_and_train.factor_baseline import build_factor_only_signals
from model_and_train.panel_rolling import run_panel_rolling
from model_and_train.pipeline import _build_model_config, _load_panel_frames, build_labeled_panel
from model_and_train.portfolio_backtest import PortfolioBacktest
from model_and_train.rolling import rolling_windows
from model_and_train.factor_library import add_common_factors
from research.model_comparison.indicator_auxiliary_experiment import (
    EXPERIMENT_MODES,
    build_model_config,
    get_experiment_spec,
)


def flatten_config(raw: dict) -> dict:
    return {
        f"{section}.{key}": value
        for section, values in raw.items()
        if isinstance(values, dict)
        for key, value in values.items()
    }


def prepare_indicator_panel(panel: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Attach the causal indicator columns required by the three model arms."""
    return add_common_factors(panel, benchmark_frame=benchmark)


def replay_factor_only(
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    windows: Iterable,
    top_k: int,
    cost_rate: float,
    hold_days: int,
) -> pd.DataFrame:
    signals = build_factor_only_signals(
        panel, top_k=top_k, mode="common", benchmark_frame=benchmark
    )
    prices = panel[["trade_date", "symbol", "open", "close", "is_suspend"]]
    rows = []
    for window in windows:
        scoped = signals[
            signals["trade_date"].between(window.test_dates[0], window.test_dates[-1])
        ]
        report = PortfolioBacktest(
            cost_rate=cost_rate, hold_days=hold_days, label_threshold=0.0
        ).run(prices, scoped, end_date=window.test_dates[-1])
        rows.append({
            "experiment_mode": "factor_only",
            "window": window.index,
            "train_start": window.train_dates[0],
            "train_end": window.train_dates[-1],
            "val_start": window.val_dates[0],
            "val_end": window.val_dates[-1],
            "test_start": window.test_dates[0],
            "test_end": window.test_dates[-1],
            "total_return_net": report.total_return_net,
            "max_drawdown": report.max_drawdown,
            "sharpe": report.sharpe,
            "calmar": report.calmar,
            "n_trades": report.n_trades,
            "indicator_auxiliary_weight": 0.0,
        })
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    summary = detail.groupby("experiment_mode", as_index=False).agg(
        mean_net_return=("total_return_net", "mean"),
        positive_windows=("total_return_net", lambda values: int((values > 0).sum())),
        n_windows=("total_return_net", "size"),
        mean_max_drawdown=("max_drawdown", "mean"),
    )
    if "rank_ic" in detail:
        rank_ic = detail.groupby("experiment_mode", as_index=False)["rank_ic"].mean()
        rank_ic = rank_ic.rename(columns={"rank_ic": "mean_rank_ic"})
        summary = summary.merge(rank_ic, on="experiment_mode", how="left")
    return summary


def run_experiment(
    modes: Iterable[str],
    max_windows: int | None = None,
    epochs: int = 20,
    seed: int = 42,
    train_days: int = 1000,
    val_days: int = 250,
    test_days: int = 250,
    step_days: int = 250,
    top_k: int = 2,
    cost_rate: float = 0.003,
    return_target_scaling: bool = False,
    ranking_loss_weight: float = 0.0,
    dispersion_loss_weight: float = 0.0,
    dispersion_target_std: float = 0.0,
    batch_mode: str = "auto",
    ranking_target: str = "raw",
    output_dir: Path = ROOT / "output" / "research" / "model_comparison" / "indicator_auxiliary_rolling",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ranking_target not in {"raw", "cross_sectional", "factor_residual"}:
        raise ValueError("ranking_target 必须是 raw、cross_sectional 或 factor_residual")
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        cfg = flatten_config(yaml.safe_load(file))
    cfg.update({
        "data.start": "2015-01-01",
        "model.look_backs": [20],
        "model.ranking_target": ranking_target,
        "model.validation_monitor": "rank_ic",
        "training.epochs": epochs,
        "training.random_seed": seed,
        "labels.label_threshold": 0.0,
        "backtest.hold_days": 10,
        "backtest.max_hold_days": 10,
        "backtest.exit_mode": "fixed",
        "ranking.factor_weight": 0.0,
        "ranking.model_weight": 1.0,
    })
    frames, benchmark = _load_panel_frames(cfg)
    panel = build_labeled_panel(
        frames, benchmark_frame=benchmark, label_threshold=0.0,
        hold_days=10, date_mode="union", extended_features=False,
    )
    panel = prepare_indicator_panel(panel, benchmark)
    dates = sorted(panel.loc[panel["valid_sample"].astype(bool), "trade_date"].unique())
    windows = list(rolling_windows(
        dates, train_days=train_days, val_days=val_days,
        test_days=test_days, embargo_days=3, step_days=step_days,
    ))
    if max_windows is not None:
        windows = windows[:max(1, max_windows)]
    output_dir.mkdir(parents=True, exist_ok=True)
    details = []
    for mode in modes:
        spec = get_experiment_spec(mode)
        if spec.factor_only:
            report = replay_factor_only(panel, benchmark, windows, top_k, cost_rate, 10)
        else:
            model_cfg = build_model_config(
                _build_model_config(cfg), mode=mode,
                return_target_scaling=return_target_scaling,
                ranking_loss_weight=ranking_loss_weight,
                dispersion_loss_weight=dispersion_loss_weight,
                dispersion_target_std=dispersion_target_std,
                batch_mode=batch_mode,
                ranking_target=ranking_target,
            )
            model_cfg.epochs = epochs
            model_cfg.random_seed = seed
            model_dir = output_dir / mode / "models"
            result = run_panel_rolling(
                panel, windows, look_back=20,
                universe_symbols=list(cfg["universe.symbols"]),
                config=model_cfg, output_dir=str(model_dir), top_k=top_k,
                cost_rate=cost_rate, hold_days=10, label_threshold=0.0,
                feature_columns=list(spec.feature_columns),
                factor_weight=0.0, model_weight=1.0,
                verbose=0, exit_mode="fixed", experiment_mode=mode,
            )
            report = pd.DataFrame(result)
        report["experiment_mode"] = mode
        report["seed"] = seed
        report["train_days"] = train_days
        report["top_k"] = top_k
        report["cost_rate"] = cost_rate
        (output_dir / mode).mkdir(parents=True, exist_ok=True)
        report.to_csv(output_dir / mode / "per_window.csv", index=False, encoding="utf-8-sig")
        details.append(report)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary = summarize(detail)
    detail.to_csv(output_dir / "per_window.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "experiment_config.json").write_text(
        json.dumps({"modes": list(modes), "seed": seed, "train_days": train_days,
                    "val_days": val_days, "test_days": test_days, "step_days": step_days,
                    "top_k": top_k, "cost_rate": cost_rate,
                    "return_target_scaling": return_target_scaling,
                    "dispersion_loss_weight": dispersion_loss_weight,
                    "dispersion_target_std": dispersion_target_std,
                    "batch_mode": batch_mode,
                    "ranking_target": ranking_target,
                    "ranking_loss_weight": ranking_loss_weight},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=EXPERIMENT_MODES, default=list(EXPERIMENT_MODES))
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-days", type=int, default=1000)
    parser.add_argument("--val-days", type=int, default=250)
    parser.add_argument("--test-days", type=int, default=250)
    parser.add_argument("--step-days", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--cost-rate", type=float, default=0.003)
    parser.add_argument("--return-target-scaling", action="store_true")
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--dispersion-loss-weight", type=float, default=0.0)
    parser.add_argument("--dispersion-target-std", type=float, default=0.0)
    parser.add_argument("--batch-mode", choices=["auto", "random", "date_grouped"], default="auto")
    parser.add_argument("--ranking-target", choices=["raw", "cross_sectional", "factor_residual"], default="raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "research" / "model_comparison" / "indicator_auxiliary_rolling")
    args = parser.parse_args()
    _, summary = run_experiment(
        args.modes, args.max_windows, args.epochs, args.seed,
        train_days=args.train_days, val_days=args.val_days,
        test_days=args.test_days, step_days=args.step_days,
        top_k=args.top_k, cost_rate=args.cost_rate,
        return_target_scaling=args.return_target_scaling,
        ranking_loss_weight=args.ranking_loss_weight,
        dispersion_loss_weight=args.dispersion_loss_weight,
        dispersion_target_std=args.dispersion_target_std,
        batch_mode=args.batch_mode,
        ranking_target=args.ranking_target,
        output_dir=args.output_dir,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
