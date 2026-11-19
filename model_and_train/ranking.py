"""Cross-sectional ranking and Top-K signal selection."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_and_select(
    predictions: pd.DataFrame,
    top_k: int = 2,
    score_column: str = "pred_return",
    min_event_probability: float | None = None,
    min_pred_return: float | None = None,
    score_weights: dict[str, float] | None = None,
    correlation_matrix: pd.DataFrame | None = None,
    correlation_threshold: float = 0.85,
) -> pd.DataFrame:
    """Rank each trade date and assign equal weights to selected symbols.

    ``score_weights`` enables a cross-sectional percentile-normalized mixture
    of prediction columns. Correlation filtering is applied greedily in score
    order, retaining the higher-scoring candidate when two symbols are too
    correlated.
    """
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    required = {"trade_date", "symbol", score_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions 缺少字段: {sorted(missing)}")
    out = predictions.copy()
    if min_event_probability is not None:
        if "event_probability" not in out.columns:
            raise ValueError("设置 min_event_probability 时必须提供 event_probability")
        out = out[out["event_probability"] >= min_event_probability].copy()
    if min_pred_return is not None:
        if "pred_return" not in out.columns:
            raise ValueError("设置 min_pred_return 时必须提供 pred_return")
        out = out[out["pred_return"] >= min_pred_return].copy()

    if score_weights is None:
        if score_column not in out.columns:
            raise ValueError(f"predictions 缺少字段: {score_column}")
        out["score"] = out[score_column].astype(float)
    else:
        if not score_weights or any(weight < 0 for weight in score_weights.values()):
            raise ValueError("score_weights 必须非空且权重不能为负")
        total_weight = float(sum(score_weights.values()))
        if total_weight <= 0:
            raise ValueError("score_weights 总和必须大于 0")
        out["score"] = 0.0
        for column, weight in score_weights.items():
            if column not in out.columns:
                raise ValueError(f"混合评分缺少字段: {column}")
            normalized = out.groupby("trade_date")[column].transform(
                lambda s: s.rank(method="average", pct=True)
            )
            out["score"] += float(weight) * normalized / total_weight

    out["rank"] = out.groupby("trade_date")["score"].rank(
        method="first", ascending=False
    )
    out["selected"] = False
    for trade_date, group in out.groupby("trade_date", sort=False):
        selected_symbols: list[str] = []
        for idx in group.sort_values(["score", "symbol"], ascending=[False, True]).index:
            symbol = out.at[idx, "symbol"]
            if correlation_matrix is not None and selected_symbols:
                if symbol in correlation_matrix.index:
                    correlations = [
                        float(correlation_matrix.loc[symbol, chosen])
                        for chosen in selected_symbols
                        if chosen in correlation_matrix.columns
                    ]
                    if correlations and max(abs(value) for value in correlations) >= correlation_threshold:
                        continue
            out.at[idx, "selected"] = True
            selected_symbols.append(symbol)
            if len(selected_symbols) >= top_k:
                break
    counts = out.groupby("trade_date")["selected"].transform("sum")
    out["weight"] = np.where(out["selected"], 1.0 / counts.replace(0, np.nan), 0.0)
    return out.sort_values(["trade_date", "rank", "symbol"]).reset_index(drop=True)


def fit_score_weights(
    validation_predictions: pd.DataFrame,
    candidates: list[dict[str, float]] | None = None,
) -> dict[str, float]:
    """Select mixture weights using validation data only.

    The function intentionally accepts a validation-named frame and rejects a
    mixed split frame when a ``split`` column is present. This keeps weight
    fitting out of the test path by construction.
    """
    required = {"trade_date", "future_return", "pred_return", "event_probability"}
    missing = required - set(validation_predictions.columns)
    if missing:
        raise ValueError(f"validation_predictions 缺少字段: {sorted(missing)}")
    if "split" in validation_predictions.columns:
        splits = set(validation_predictions["split"].dropna().astype(str))
        if splits - {"val", "validation"}:
            raise ValueError("fit_score_weights 只能使用验证集数据")
    frame = validation_predictions.copy()
    candidates = candidates or [
        {"pred_return": 1.0, "event_probability": 0.0},
        {"pred_return": 0.75, "event_probability": 0.25},
        {"pred_return": 0.5, "event_probability": 0.5},
        {"pred_return": 0.25, "event_probability": 0.75},
        {"pred_return": 0.0, "event_probability": 1.0},
    ]
    best = None
    for candidate in candidates:
        if not candidate or any(float(value) < 0 for value in candidate.values()):
            raise ValueError("候选评分权重必须非空且不能为负")
        total = float(sum(candidate.values()))
        if total <= 0:
            raise ValueError("候选评分权重总和必须大于 0")
        score = pd.Series(0.0, index=frame.index)
        for column, value in candidate.items():
            if column not in frame.columns:
                raise ValueError(f"候选评分权重缺少字段: {column}")
            percentile = frame.groupby("trade_date")[column].rank(
                method="average", pct=True
            )
            score = score + float(value) / total * percentile
        temp = pd.DataFrame({"score": score, "future_return": frame["future_return"]})
        def _daily_score_ic(group: pd.DataFrame) -> float | None:
            if group["score"].nunique() < 2 or group["future_return"].nunique() < 2:
                return None
            return float(group["score"].corr(group["future_return"], method="spearman"))

        daily_ic = temp.groupby(frame["trade_date"]).apply(_daily_score_ic)
        valid = daily_ic.dropna()
        objective = float(valid.mean()) if len(valid) else float("-inf")
        if best is None or objective > best[0]:
            best = (objective, {key: float(value) / total for key, value in candidate.items()})
    return best[1]
