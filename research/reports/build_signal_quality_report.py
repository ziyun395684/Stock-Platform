"""Build daily cross-sectional signal quality summaries from predictions."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_and_train.rolling import classify_market_regime


def _normalize_trade_dates(values: pd.Series) -> pd.Series:
    """Normalize YYYY-MM-DD and YYYYMMDD inputs to comparable integers."""
    numeric = pd.to_numeric(values, errors="coerce")
    parsed = pd.to_datetime(values, errors="coerce")
    normalized = numeric.where(numeric.notna(), parsed.dt.strftime("%Y%m%d").astype(float))
    return normalized.round().astype("Int64")


def _daily_rank_ic(group: pd.DataFrame) -> float:
    if (
        len(group) < 2
        or group["pred_return"].nunique() < 2
        or group["future_return"].nunique() < 2
    ):
        return np.nan
    return float(group["pred_return"].rank().corr(group["future_return"].rank()))


def summarize_signal_quality(frame: pd.DataFrame, top_k: int = 2) -> dict:
    required = {"trade_date", "symbol", "pred_return", "future_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"预测数据缺少字段: {sorted(missing)}")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    data = frame[list(required)].copy()
    data["pred_return"] = pd.to_numeric(data["pred_return"], errors="coerce")
    data["future_return"] = pd.to_numeric(data["future_return"], errors="coerce")
    data = data.dropna(subset=["pred_return", "future_return"])
    if data.empty:
        return {
            "n_dates": 0, "mean_rank_ic": np.nan, "top_k_hit_rate": np.nan,
            "top_k_mean_future_return": np.nan, "collapsed_dates": 0,
        }

    daily_ics = pd.Series(
        {
            date: _daily_rank_ic(group)
            for date, group in data.groupby("trade_date", sort=True)
        }
    )
    selected = data.sort_values(["trade_date", "pred_return"], ascending=[True, False])
    selected = selected.groupby("trade_date", sort=True).head(top_k)
    return {
        "n_dates": int(data["trade_date"].nunique()),
        "mean_rank_ic": float(daily_ics.mean()),
        "top_k_hit_rate": float((selected["future_return"] >= 0).mean()),
        "top_k_mean_future_return": float(selected["future_return"].mean()),
        "collapsed_dates": int(
            data.groupby("trade_date")["pred_return"].nunique().le(1).sum()
        ),
    }


def build_daily_report(frame: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    """Return one signal-quality row per trade date for detailed diagnosis."""
    required = {"trade_date", "symbol", "pred_return", "future_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"预测数据缺少字段: {sorted(missing)}")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    data = frame[list(required)].copy()
    data["pred_return"] = pd.to_numeric(data["pred_return"], errors="coerce")
    data["future_return"] = pd.to_numeric(data["future_return"], errors="coerce")
    data = data.dropna(subset=["pred_return", "future_return"])
    rows = []
    for date, group in data.groupby("trade_date", sort=True):
        selected = group.sort_values("pred_return", ascending=False).head(top_k)
        rows.append({
            "trade_date": date,
            "n_symbols": int(len(group)),
            "rank_ic": _daily_rank_ic(group),
            "top_k_hit_rate": float((selected["future_return"] >= 0).mean()),
            "top_k_mean_future_return": float(selected["future_return"].mean()),
            "prediction_collapsed": bool(group["pred_return"].nunique() <= 1),
        })
    return pd.DataFrame(rows, columns=[
        "trade_date", "n_symbols", "rank_ic", "top_k_hit_rate",
        "top_k_mean_future_return", "prediction_collapsed",
    ])


def build_symbol_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-ETF sample and realized-return diagnostics."""
    required = {"trade_date", "symbol", "pred_return", "future_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"预测数据缺少字段: {sorted(missing)}")
    data = frame[list(required)].copy()
    data["pred_return"] = pd.to_numeric(data["pred_return"], errors="coerce")
    data["future_return"] = pd.to_numeric(data["future_return"], errors="coerce")
    data = data.dropna(subset=["pred_return", "future_return"])
    rows = []
    for symbol, group in data.groupby("symbol", sort=True):
        rows.append({
            "symbol": symbol,
            "n_samples": int(len(group)),
            "n_dates": int(group["trade_date"].nunique()),
            "mean_rank_ic": _daily_rank_ic(group) if len(group) >= 2 else np.nan,
            "mean_pred_return": float(group["pred_return"].mean()),
            "mean_future_return": float(group["future_return"].mean()),
            "positive_future_return_rate": float((group["future_return"] >= 0).mean()),
        })
    return pd.DataFrame(rows, columns=[
        "symbol", "n_samples", "n_dates", "mean_rank_ic", "mean_pred_return",
        "mean_future_return", "positive_future_return_rate",
    ])


def attach_market_regime(
    daily_report: pd.DataFrame,
    benchmark: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Attach causal benchmark regimes to daily signal rows."""
    if lookback <= 0:
        raise ValueError("lookback 必须大于 0")
    required_daily = {"trade_date"}
    missing_daily = required_daily - set(daily_report.columns)
    if missing_daily:
        raise ValueError(f"日度报告缺少字段: {sorted(missing_daily)}")
    benchmark = benchmark.rename(columns={"date": "trade_date"})
    missing_benchmark = {"trade_date", "close"} - set(benchmark.columns)
    if missing_benchmark:
        raise ValueError(f"基准数据缺少字段: {sorted(missing_benchmark)}")

    bench = benchmark[["trade_date", "close"]].copy()
    bench["trade_date"] = _normalize_trade_dates(bench["trade_date"])
    bench["close"] = pd.to_numeric(bench["close"], errors="coerce")
    bench = bench.dropna().sort_values("trade_date")
    bench["benchmark_return_1"] = bench["close"].pct_change(fill_method=None)
    bench["volatility_20"] = bench["benchmark_return_1"].rolling(
        lookback, min_periods=min(5, lookback)
    ).std()

    out = daily_report.copy()
    regimes = []
    normalized_dates = _normalize_trade_dates(out["trade_date"])
    for date in normalized_dates:
        history = bench[bench["trade_date"] <= date].tail(lookback)
        if history.empty:
            regimes.append("unknown")
        else:
            regimes.append(classify_market_regime(
                history["benchmark_return_1"], history["volatility_20"]
            ))
    out["market_regime"] = regimes
    return out


def build_report(prediction_paths: list[str], top_k: int = 2) -> pd.DataFrame:
    rows = []
    for value in prediction_paths:
        path = Path(value)
        frame = pd.read_csv(path)
        metrics = summarize_signal_quality(frame, top_k=top_k)
        metrics["source"] = str(path)
        rows.append(metrics)
    return pd.DataFrame(rows)


def _prediction_metadata(path: str) -> dict:
    value = str(path).replace("\\", "/")
    train_match = re.search(r"target_[^/]+_(\d+)", value)
    residual_train_match = re.search(r"residual_rolling_(\d+)(?:_[^/]*)?", value)
    window_match = re.search(r"rolling_panel_\d+_w(\d+)_predictions\.csv$", value)
    target_match = re.search(r"target_([^/]+)_\d+", value)
    mode_match = re.search(r"/([^/]+)/models/rolling_panel_", value)
    train_days = (
        int(train_match.group(1))
        if train_match
        else int(residual_train_match.group(1))
        if residual_train_match
        else np.nan
    )
    return {
        "train_days": train_days,
        "window": int(window_match.group(1)) if window_match else np.nan,
        "target": target_match.group(1) if target_match else "factor_residual" if residual_train_match else "unknown",
        "experiment_mode": mode_match.group(1) if mode_match else "factor_residual" if residual_train_match else "unknown",
    }


def build_acceptance_report(
    frames: list[tuple[str, pd.DataFrame]],
    top_k: int = 2,
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one normalized signal-quality row per rolling test prediction."""
    rows = []
    for path, frame in frames:
        metrics = summarize_signal_quality(frame, top_k=top_k)
        daily = build_daily_report(frame, top_k=top_k)
        if benchmark is not None:
            daily = attach_market_regime(daily, benchmark)
        metadata = _prediction_metadata(path)
        metrics.update(metadata)
        metrics["n_symbols"] = int(frame["symbol"].nunique())
        metrics["collapsed_date_ratio"] = (
            float(daily["prediction_collapsed"].mean()) if len(daily) else np.nan
        )
        metrics["market_regime"] = (
            daily["market_regime"].mode().iloc[0]
            if "market_regime" in daily and not daily["market_regime"].empty
            else "unknown"
        )
        metrics["source"] = str(path)
        rows.append(metrics)
    columns = [
        "train_days", "window", "target", "experiment_mode", "n_dates",
        "n_symbols", "mean_rank_ic", "top_k_hit_rate", "top_k_mean_future_return",
        "collapsed_dates", "collapsed_date_ratio", "source",
        "market_regime",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_regime_report(
    frames: list[tuple[str, pd.DataFrame]],
    benchmark: pd.DataFrame,
    top_k: int = 2,
    lookback: int = 20,
) -> pd.DataFrame:
    """Aggregate daily signal quality by causal market regime."""
    if benchmark is None:
        raise ValueError("生成市场阶段信号质量报告必须提供 benchmark")
    rows = []
    for path, frame in frames:
        daily = build_daily_report(frame, top_k=top_k)
        daily = attach_market_regime(daily, benchmark, lookback=lookback)
        metadata = _prediction_metadata(path)
        daily["train_days"] = metadata["train_days"]
        daily["window"] = metadata["window"]
        daily["source"] = str(path)
        rows.append(daily)
    if not rows:
        return pd.DataFrame(
            columns=[
                "train_days", "market_regime", "n_dates", "mean_rank_ic",
                "top_k_hit_rate", "top_k_mean_future_return",
                "collapsed_date_ratio", "n_windows",
            ]
        )
    data = pd.concat(rows, ignore_index=True)
    result = (
        data.groupby(["train_days", "market_regime"], dropna=False, as_index=False)
        .agg(
            n_dates=("trade_date", "nunique"),
            mean_rank_ic=("rank_ic", "mean"),
            top_k_hit_rate=("top_k_hit_rate", "mean"),
            top_k_mean_future_return=("top_k_mean_future_return", "mean"),
            collapsed_date_ratio=("prediction_collapsed", "mean"),
            n_windows=("source", "nunique"),
        )
        .sort_values(["train_days", "market_regime"], na_position="last")
    )
    return result


def assess_regime_coverage(
    regime_report: pd.DataFrame,
    min_dates: int = 20,
    min_windows: int = 2,
) -> pd.DataFrame:
    """Mark whether each train-window/regime row has enough evidence."""
    if min_dates <= 0 or min_windows <= 0:
        raise ValueError("min_dates 和 min_windows 必须大于 0")
    required = {"train_days", "market_regime", "n_dates", "n_windows"}
    missing = required - set(regime_report.columns)
    if missing:
        raise ValueError(f"阶段报告缺少字段: {sorted(missing)}")

    result = regime_report.copy()
    reasons = []
    sufficient = []
    for _, row in result.iterrows():
        row_reasons = []
        if int(row["n_dates"]) < min_dates:
            row_reasons.append(f"dates<{min_dates}")
        if int(row["n_windows"]) < min_windows:
            row_reasons.append(f"windows<{min_windows}")
        sufficient.append(not row_reasons)
        reasons.append("ok" if not row_reasons else ";".join(row_reasons))
    result["coverage_sufficient"] = sufficient
    result["coverage_reason"] = reasons
    result["min_dates"] = min_dates
    result["min_windows"] = min_windows
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_paths", nargs="+")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--output",
        default="output/research/reports/signal_quality_summary.csv",
    )
    parser.add_argument(
        "--daily-output",
        default=None,
        help="可选：同时输出逐交易日信号质量明细",
    )
    parser.add_argument(
        "--symbol-output",
        default=None,
        help="可选：同时输出逐 ETF 信号质量明细",
    )
    parser.add_argument(
        "--acceptance-output",
        default=None,
        help="可选：输出带训练窗口/滚动窗口/目标模式元数据的统一验收明细",
    )
    parser.add_argument(
        "--benchmark-path",
        default=None,
        help="可选：基准 ETF 价格 CSV，用于因果市场阶段标注",
    )
    parser.add_argument(
        "--regime-output",
        default=None,
        help="可选：按市场阶段汇总信号质量；必须同时提供 --benchmark-path",
    )
    parser.add_argument(
        "--coverage-output",
        default=None,
        help="可选：输出市场阶段日期/窗口覆盖门禁；必须同时提供 --regime-output",
    )
    parser.add_argument("--min-regime-dates", type=int, default=20)
    parser.add_argument("--min-regime-windows", type=int, default=2)
    args = parser.parse_args()
    report = build_report(args.prediction_paths, top_k=args.top_k)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print(f"saved: {output}")
    if args.daily_output:
        daily_frames = []
        for value in args.prediction_paths:
            daily = build_daily_report(pd.read_csv(value), top_k=args.top_k)
            daily["source"] = str(value)
            daily_frames.append(daily)
        daily_output = ROOT / args.daily_output
        daily_output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(daily_frames, ignore_index=True).to_csv(
            daily_output, index=False, encoding="utf-8-sig"
        )
        print(f"saved: {daily_output}")
    if args.symbol_output:
        symbol_frames = []
        for value in args.prediction_paths:
            symbol = build_symbol_report(pd.read_csv(value))
            symbol["source"] = str(value)
            symbol_frames.append(symbol)
        symbol_output = ROOT / args.symbol_output
        symbol_output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(symbol_frames, ignore_index=True).to_csv(
            symbol_output, index=False, encoding="utf-8-sig"
        )
        print(f"saved: {symbol_output}")
    if args.acceptance_output:
        acceptance = build_acceptance_report(
            [(value, pd.read_csv(value)) for value in args.prediction_paths],
            top_k=args.top_k,
            benchmark=pd.read_csv(args.benchmark_path) if args.benchmark_path else None,
        )
        acceptance_output = ROOT / args.acceptance_output
        acceptance_output.parent.mkdir(parents=True, exist_ok=True)
        acceptance.to_csv(acceptance_output, index=False, encoding="utf-8-sig")
        print(f"saved: {acceptance_output}")
    if args.regime_output:
        if not args.benchmark_path:
            raise ValueError("--regime-output 必须同时提供 --benchmark-path")
        regime = build_regime_report(
            [(value, pd.read_csv(value)) for value in args.prediction_paths],
            benchmark=pd.read_csv(args.benchmark_path),
            top_k=args.top_k,
        )
        regime_output = ROOT / args.regime_output
        regime_output.parent.mkdir(parents=True, exist_ok=True)
        regime.to_csv(regime_output, index=False, encoding="utf-8-sig")
        print(f"saved: {regime_output}")
        if args.coverage_output:
            coverage = assess_regime_coverage(
                regime,
                min_dates=args.min_regime_dates,
                min_windows=args.min_regime_windows,
            )
            coverage_output = ROOT / args.coverage_output
            coverage_output.parent.mkdir(parents=True, exist_ok=True)
            coverage.to_csv(coverage_output, index=False, encoding="utf-8-sig")
            print(f"saved: {coverage_output}")
    elif args.coverage_output:
        raise ValueError("--coverage-output 必须同时提供 --regime-output")


if __name__ == "__main__":
    main()
