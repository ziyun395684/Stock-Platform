"""Build one comparable matrix from rolling model summary CSV files."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "source",
    "model",
    "train_days",
    "mean_net_return",
    "positive_windows",
    "n_windows",
    "mean_max_drawdown",
    "mean_auc",
    "mean_rank_ic",
    "summary_path",
]
REQUIRED_COLUMNS = {
    "mean_net_return",
    "positive_windows",
    "n_windows",
    "mean_max_drawdown",
}


def _train_days(frame: pd.DataFrame, path: Path) -> int | None:
    if "train_days" in frame.columns:
        value = pd.to_numeric(frame["train_days"].iloc[0], errors="coerce")
        if pd.notna(value):
            return int(value)
    match = re.search(r"(?:comparison_|blend_|rolling_|target_[^\\/]+_)(\d+)", str(path))
    if not match:
        match = re.search(r"(?:^|[\\/])[^\\/]*_(\d+)(?:[\\/]|$)", str(path.parent))
    return int(match.group(1)) if match else None


def _normalize(frame: pd.DataFrame, path: Path, root: Path) -> pd.DataFrame:
    if frame.empty or not REQUIRED_COLUMNS.issubset(frame.columns):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    rows = []
    days = _train_days(frame, path)
    for _, row in frame.iterrows():
        model = str(row.get("model", ""))
        if not model or model == "nan":
            model = str(row.get("experiment_mode", ""))
        if "model_weight" in frame.columns:
            model = f"Blend model_weight={float(row['model_weight']):g}"
        rows.append(
            {
                "source": path.parent.parent.name if path.parent.name == "models" else path.parent.name,
                "model": model,
                "train_days": days,
                "mean_net_return": pd.to_numeric(row["mean_net_return"], errors="coerce"),
                "positive_windows": pd.to_numeric(row["positive_windows"], errors="coerce"),
                "n_windows": pd.to_numeric(row["n_windows"], errors="coerce"),
                "mean_max_drawdown": pd.to_numeric(row["mean_max_drawdown"], errors="coerce"),
                "mean_auc": pd.to_numeric(row.get("mean_auc"), errors="coerce"),
                "mean_rank_ic": pd.to_numeric(row.get("mean_rank_ic"), errors="coerce"),
                "summary_path": str(path),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def build_comparison_matrix(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    frames = []
    for path in sorted(root.glob("**/summary.csv")):
        normalized = _normalize(pd.read_csv(path), path, root)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["train_days", "source", "model"], na_position="last").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output/research/model_comparison")
    parser.add_argument("--output", default="output/research/reports/model_comparison_matrix.csv")
    args = parser.parse_args()
    result = build_comparison_matrix(args.root)
    if result.empty:
        raise SystemExit("没有找到字段完整的模型汇总文件")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
