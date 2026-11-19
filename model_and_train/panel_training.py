"""Training orchestration for one shared multi-ETF multitask model."""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.panel_sequences import PanelSequenceResult

from .config import ModelConfig
from .multitask_model import MultiTaskLSTMAttentionModel

logger = logging.getLogger(__name__)


def fit_indicator_target_scaler(sample_meta, indices: np.ndarray, columns) -> StandardScaler:
    """Fit indicator-target scaling statistics on training samples only."""
    values = build_indicator_targets(sample_meta, indices, columns)
    scaler = StandardScaler()
    scaler.fit(values)
    scaler.scale_ = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    return scaler


def fit_return_target_scaler(sample_meta, indices: np.ndarray) -> StandardScaler:
    """Fit return-target scaling statistics on training samples only."""
    values = build_return_targets(sample_meta, indices, "raw").reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(values)
    scaler.scale_ = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    return scaler


def build_indicator_targets(
    sample_meta, indices: np.ndarray, columns, scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """Select causal indicator targets aligned with panel samples."""
    columns = list(columns or [])
    if not columns:
        raise ValueError("指标目标列不能为空")
    missing = [column for column in columns if column not in sample_meta.columns]
    if missing:
        raise ValueError(f"指标目标列缺失: {missing}")
    values = sample_meta.iloc[np.asarray(indices, dtype=int)][columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype="float32")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is not None:
        values = scaler.transform(values)
    return np.asarray(values, dtype="float32")


def build_return_targets(
    sample_meta,
    indices: np.ndarray,
    target_mode: str = "raw",
    return_clip: float = 0.2,
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    """Build regression targets without changing the stored raw returns."""
    if target_mode not in {"raw", "cross_sectional", "factor_residual"}:
        raise ValueError("target_mode 必须为 raw、cross_sectional 或 factor_residual")
    indices = np.asarray(indices, dtype=int)
    raw = sample_meta["future_return"].to_numpy(dtype="float32")
    if target_mode == "raw":
        target = raw[indices]
    elif target_mode == "factor_residual":
        teacher = _factor_teacher_scores(sample_meta, np.arange(len(sample_meta)), return_clip)
        if teacher is None:
            raise ValueError("factor_residual 需要 sample_meta 包含 legacy_factor_score")
        target = (raw - teacher)[indices]
    else:
        frame = sample_meta[["trade_date", "future_return"]].copy()
        frame["future_return"] = frame["future_return"].astype("float32")
        frame["target"] = frame["future_return"] - frame.groupby("trade_date")[
            "future_return"
        ].transform("mean")
        target = frame["target"].to_numpy(dtype="float32")[indices]
    if scaler is not None:
        target = scaler.transform(np.asarray(target, dtype="float32").reshape(-1, 1)).reshape(-1)
    return np.asarray(target, dtype="float32")


def _factor_teacher_scores(
    sample_meta,
    indices: np.ndarray,
    return_clip: float,
) -> Optional[np.ndarray]:
    """Convert causal legacy scores to date-wise percentile teacher targets."""
    if "legacy_factor_score" not in sample_meta.columns:
        return None
    frame = sample_meta[["trade_date", "legacy_factor_score"]].copy()
    frame["legacy_factor_score"] = np.asarray(
        frame["legacy_factor_score"], dtype="float32"
    )
    frame["factor_teacher"] = frame.groupby("trade_date")[
        "legacy_factor_score"
    ].rank(method="average", pct=True)
    # Map the percentile teacher into the regression head's return scale.
    frame["factor_teacher"] = (
        frame["factor_teacher"] - 0.5
    ) * 2.0 * float(return_clip)
    return frame.iloc[indices]["factor_teacher"].to_numpy(dtype="float32")


def restore_return_predictions(
    predictions: dict,
    sample_meta,
    indices: np.ndarray,
    target_mode: str,
    return_clip: float,
    return_target_scaler: Optional[dict] = None,
) -> dict:
    """Restore model outputs to return scale after residual-target training."""
    restored = dict(predictions)
    if return_target_scaler is not None:
        mean = np.asarray(return_target_scaler["mean"], dtype="float32")
        scale = np.asarray(return_target_scaler["scale"], dtype="float32")
        restored["pred_return"] = (
            np.asarray(restored["pred_return"], dtype="float32") * scale + mean
        )
    if target_mode != "factor_residual":
        return restored
    teacher = _factor_teacher_scores(sample_meta, np.asarray(indices, dtype=int), return_clip)
    if teacher is None:
        raise ValueError("factor_residual 需要 sample_meta 包含 legacy_factor_score")
    restored["pred_return"] = np.asarray(restored["pred_return"], dtype="float32") + teacher
    return restored


def panel_sample_statistics(sample_meta) -> list[dict]:
    """Return per-symbol/per-split sample and positive-label statistics."""
    required = {"symbol", "split", "event_label"}
    missing = required - set(sample_meta.columns)
    if missing:
        raise ValueError(f"sample_meta 缺少字段: {sorted(missing)}")
    grouped = sample_meta.groupby(["symbol", "split"], sort=True)
    rows = []
    for (symbol, split), group in grouped:
        total = int(len(group))
        positive = int(group["event_label"].sum())
        rows.append({
            "symbol": symbol,
            "split": split,
            "samples": total,
            "positive": positive,
            "positive_ratio": positive / total if total else 0.0,
        })
    return rows


def warn_on_panel_samples(
    sample_meta,
    min_positive_count: int = 50,
    min_positive_ratio: float = 0.05,
    min_sample_count: int = 50,
) -> list[str]:
    """Warn when a symbol/split group has too few positive labels."""
    warnings = []
    for row in panel_sample_statistics(sample_meta):
        if (
            row["samples"] < min_sample_count
            or
            row["positive"] < min_positive_count
            or row["positive_ratio"] < min_positive_ratio
        ):
            message = (
                f"面板样本不足: symbol={row['symbol']} split={row['split']} "
                f"samples={row['samples']} positive={row['positive']} "
                f"ratio={row['positive_ratio']:.4f} "
                f"thresholds=(samples={min_sample_count}, positive={min_positive_count}, "
                f"ratio={min_positive_ratio})"
            )
            logger.warning(message)
            warnings.append(message)
    return warnings


def train_panel_model(
    seq_res: PanelSequenceResult,
    universe_symbols: Sequence[str],
    config: Optional[ModelConfig] = None,
    output_path: Optional[str] = None,
    label_config: Optional[dict] = None,
    verbose: int = 0,
) -> dict:
    """Train one shared model using date-isolated panel indices."""
    train_idx = np.asarray(seq_res.train_index, dtype=int)
    val_idx = np.asarray(seq_res.val_index, dtype=int)
    if len(train_idx) == 0:
        raise ValueError("面板训练集没有有效样本")

    model = MultiTaskLSTMAttentionModel(
        look_back=seq_res.X.shape[1],
        feature_dim=seq_res.X.shape[2],
        config=config,
        feature_names=seq_res.feature_names,
        universe_symbols=list(universe_symbols),
    )
    model.scaler_metadata = {
        "feature_names": list(seq_res.feature_names),
        "mean": seq_res.scaler.mean_.tolist(),
        "scale": seq_res.scaler.scale_.tolist(),
    }
    model.label_config = dict(label_config or {"hold_days": 3, "threshold": 0.05})
    target_mode = getattr(model.config, "ranking_target", "raw")
    model.label_config["ranking_target"] = target_mode
    model.model_version = "panel-multitask-v1"
    warn_on_panel_samples(
        seq_res.sample_meta,
        min_positive_count=model.config.min_positive_count,
        min_positive_ratio=model.config.min_positive_ratio,
        min_sample_count=model.config.min_sample_count,
    )
    y = {"future_return": seq_res.y_return, "event_label": seq_res.y_event}
    return_target_scaler = None
    if model.config.return_target_scaling:
        return_target_scaler = fit_return_target_scaler(seq_res.sample_meta, train_idx)
        model.label_config["return_target_scaler"] = {
            "mean": return_target_scaler.mean_.tolist(),
            "scale": return_target_scaler.scale_.tolist(),
        }
    y_train_return = build_return_targets(
        seq_res.sample_meta, train_idx, target_mode, model.config.return_clip,
        scaler=return_target_scaler,
    )
    y_val_return = build_return_targets(
        seq_res.sample_meta, val_idx, target_mode, model.config.return_clip,
        scaler=return_target_scaler,
    )
    factor_teacher = None if target_mode == "factor_residual" else _factor_teacher_scores(
        seq_res.sample_meta, train_idx, model.config.return_clip
    )
    factor_teacher_val = None if target_mode == "factor_residual" else _factor_teacher_scores(
        seq_res.sample_meta, val_idx, model.config.return_clip
    )
    indicator_targets = None
    indicator_targets_val = None
    if model.config.indicator_auxiliary_weight > 0.0:
        indicator_scaler = fit_indicator_target_scaler(
            seq_res.sample_meta, train_idx, model.config.indicator_target_columns
        )
        indicator_targets = build_indicator_targets(
            seq_res.sample_meta,
            train_idx,
            model.config.indicator_target_columns,
            scaler=indicator_scaler,
        )
        indicator_targets_val = build_indicator_targets(
            seq_res.sample_meta,
            val_idx,
            model.config.indicator_target_columns,
            scaler=indicator_scaler,
        ) if len(val_idx) else None
        model.label_config["indicator_target_scaler"] = {
            "columns": list(model.config.indicator_target_columns),
            "mean": indicator_scaler.mean_.tolist(),
            "scale": indicator_scaler.scale_.tolist(),
        }
    ranking_group_ids = None
    ranking_group_ids_val = None
    if model.config.ranking_loss_weight > 0.0 or getattr(model.config, "batch_mode", "auto") == "date_grouped":
        date_codes = {
            date: index
            for index, date in enumerate(
                sorted(seq_res.sample_meta["trade_date"].unique())
            )
        }
        groups = seq_res.sample_meta["trade_date"].map(date_codes).to_numpy(
            dtype="float32"
        )
        ranking_group_ids = groups[train_idx]
        ranking_group_ids_val = groups[val_idx]
    result = model.train(
        seq_res.X[train_idx],
        {"future_return": y_train_return, "event_label": y["event_label"][train_idx]},
        factor_teacher=factor_teacher,
        X_val=seq_res.X[val_idx] if len(val_idx) else None,
        y_val=(
            {"future_return": y_val_return, "event_label": y["event_label"][val_idx]}
            if len(val_idx) else None
        ),
        factor_teacher_val=factor_teacher_val,
        indicator_targets=indicator_targets,
        indicator_targets_val=indicator_targets_val,
        ranking_group_ids=ranking_group_ids,
        ranking_group_ids_val=ranking_group_ids_val,
        val_meta=seq_res.sample_meta.iloc[val_idx] if len(val_idx) else None,
        save_path=output_path,
        verbose=verbose,
    )
    return {
        **result,
        "path": output_path,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "feature_names": list(seq_res.feature_names),
        "universe_symbols": list(universe_symbols),
        "sample_statistics": panel_sample_statistics(seq_res.sample_meta),
    }
