"""Task 3.3 基线策略对比模块.

对应 Todo Task3.3 / 需求 §10.2（备选模型）与 §5.3：

    * `BaselineStrategy` 抽象基类：统一 `score(df) -> pd.Series`（得分，
      越大越看涨，**仅用截至当前行 T 的数据，无未来泄露**）与
      `evaluate(...)`（输出指标 dict，格式与 Task3.2 对齐）。
    * 实现：Momentum（20 日动量）、MACD（DIF>DEA）、Logistic（逻辑回归，
      仅在训练段拟合）。
    * 指标输出与 `classification_metrics` 对齐：accuracy/precision/recall/
      f1/auc/confusion，JSON 安全。

评估口径：
    * 信号在样本日 T 生成（仅用 T 及之前数据），与 `LabelResult` 的
      样本日期按 trade_date 对齐后计算指标，禁止下标对齐。
    * 结果可复现：Logistic 支持 random_seed；因子列确定后结果确定。
"""
from __future__ import annotations

import abc
import logging
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .evaluation import classification_metrics, to_json_safe

logger = logging.getLogger(__name__)


class BaselineStrategy(abc.ABC):
    """基线策略抽象基类（Task3.3）。"""

    #: 策略名（报表/对比表标识）。
    name: str = "baseline"

    def __init__(self, label_threshold: float = 0.05, **kwargs) -> None:
        self.label_threshold = float(label_threshold)
        self.kwargs = dict(kwargs)

    @abc.abstractmethod
    def score(self, df: pd.DataFrame) -> pd.Series:
        """对统一 Schema 日线 df（升序，含因子列）逐行打分 [0-1]。

        仅允许使用截至当前行 T 的数据（禁止未来泄露）。
        返回 Series 与 df 对齐。
        """

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """得分 >= threshold -> 预测上涨事件(1)，否则 0。"""
        return (self.score(df) >= threshold).astype(int)

    def evaluate(
        self,
        df: pd.DataFrame,
        labeled,
        threshold: float = 0.5,
    ) -> Dict[str, object]:
        """在标签样本上评估基线，指标格式与 Task3.2 对齐。

        Args:
            df: 与标签生成同一份清洗+复权日线（含因子）。
            labeled: `data.labels.LabelResult`（其 df 含 trade_date/label）。
            threshold: 正例判定阈值。

        Returns:
            dict：name + classification_metrics + n_samples 等（JSON 安全）。
        """
        score = self.score(df)
        sample_dates = labeled.df["trade_date"].to_numpy()
        labels = labeled.df["label"].to_numpy()

        date_to_pos = {d: i for i, d in enumerate(df["trade_date"].to_numpy())}
        pos = np.array([date_to_pos[d] for d in sample_dates if d in date_to_pos])
        keep = np.array([d in date_to_pos for d in sample_dates])
        if len(pos) == 0:
            raise ValueError(f"{self.name}: 标签样本日期均不在 df 中，无法评估")

        y_true = labels[keep]
        y_score = score.to_numpy()[pos]

        metrics = classification_metrics(y_true, y_score, threshold=threshold)
        return to_json_safe({"name": self.name, **metrics})


class MomentumStrategy(BaselineStrategy):
    """动量基线：20 日累计收益 > 0 视为看涨（需求§5.3 参考基线）。

    score = clip(0.5 + ret20 / 0.1, 0, 1)：ret20>0 时 score>0.5。
    """

    name = "momentum"

    def __init__(self, lookback: int = 20, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lookback = int(lookback)

    def score(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ret20 = close / close.shift(self.lookback) - 1.0
        s = 0.5 + ret20 / 0.1
        return s.clip(lower=0.0, upper=1.0).fillna(0.5)


class MACDStrategy(BaselineStrategy):
    """MACD 基线：DIF > DEA（金叉状态）视为看涨。

    score = clip(0.5 + macd_hist / (3*std20), 0, 1)。
    """

    name = "macd"

    def score(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = dif - dea
        std = hist.rolling(20).std()
        s = 0.5 + hist / (3.0 * std.replace(0, np.nan))
        return s.clip(lower=0.0, upper=1.0).fillna(0.5)


class LogisticStrategy(BaselineStrategy):
    """Logistic 回归基线：在**训练段**拟合，对评估段打分。

    score 为预测正类概率。训练段 = 前 70% 行（与数据集划分 70/15/15
    的 train 比例一致，时序切分禁 shuffle）。
    """

    name = "logistic"

    def __init__(
        self,
        feature_columns: Optional[Sequence[str]] = None,
        train_ratio: float = 0.70,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.feature_columns = list(feature_columns) if feature_columns else None
        self.train_ratio = float(train_ratio)
        self.random_seed = random_seed
        self._model = None

    def _pick_features(self, df: pd.DataFrame) -> list:
        if self.feature_columns is not None:
            missing = [c for c in self.feature_columns if c not in df.columns]
            if missing:
                raise ValueError(f"LogisticStrategy: 缺少特征列 {missing}")
            return self.feature_columns
        excluded = {
            "trade_date", "date", "open", "high", "low", "close", "volume",
            "amount", "label", "return", "buy_price", "sell_price", "weight",
            "is_suspend", "float_shares",
        }
        return [c for c in df.columns
                if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

    def fit(self, df: pd.DataFrame, y: Sequence[int]) -> "LogisticStrategy":
        """在给定训练集上拟合（特征自动选择数值列）。"""
        from sklearn.linear_model import LogisticRegression

        feats = self._pick_features(df)
        X = df[feats].ffill().fillna(0.0).to_numpy(dtype="float64")
        yy = np.asarray(y)
        if len(np.unique(yy)) < 2:
            logger.warning("LogisticStrategy: 训练标签仅单一类别，跳过拟合")
            self._model = None
            return self
        self._model = LogisticRegression(
            max_iter=1000, random_state=self.random_seed
        ).fit(X, yy)
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("LogisticStrategy: 需先 fit()")
        feats = self._pick_features(df)
        X = df[feats].ffill().fillna(0.0).to_numpy(dtype="float64")
        proba = self._model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=df.index)

    def evaluate(self, df: pd.DataFrame, labeled, threshold: float = 0.5) -> Dict:
        # 时序切分：前 70% 拟合、后 30% 评估（禁 shuffle，与阶段一口径一致）。
        n = len(df)
        cut = int(round(n * self.train_ratio))
        sample_dates = labeled.df["trade_date"].to_numpy()
        labels = labeled.df["label"].to_numpy()
        df_dates = df["trade_date"].to_numpy()
        date_to_pos = {d: i for i, d in enumerate(df_dates)}
        # 训练段内有效样本（样本日在 df 中的行号 < cut）。
        tr_rows, tr_y = [], []
        for i, d in enumerate(sample_dates):
            pos = date_to_pos.get(d)
            if pos is not None and pos < cut:
                tr_rows.append(pos)
                tr_y.append(labels[i])
        if not tr_rows:
            raise ValueError("LogisticStrategy: 训练段无标签样本，无法拟合")
        self.fit(df.iloc[tr_rows], np.asarray(tr_y))
        return super().evaluate(df, labeled, threshold=threshold)


def compare_baselines(
    df: pd.DataFrame,
    labeled,
    strategies: Optional[Sequence[BaselineStrategy]] = None,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """运行一组基线策略并输出对比表（指标格式与 Task3.2 对齐）。

    Returns:
        {"results": [{name, accuracy, precision, recall, f1, auc, ...}],
         "table": 文本表格}
    """
    strategies = strategies or [
        MomentumStrategy(),
        MACDStrategy(),
        LogisticStrategy(random_seed=0),
    ]
    results = []
    for s in strategies:
        try:
            results.append(s.evaluate(df, labeled, threshold=threshold))
        except Exception as exc:  # noqa: BLE001 - 单个基线失败不阻断对比
            logger.warning("基线 %s 评估失败：%s", s.name, exc)
            results.append({"name": s.name, "error": str(exc)})

    header = f"{'name':>10} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'auc':>6}"
    lines = [header]
    for r in results:
        if "error" in r:
            lines.append(f"{r['name']:>10}  ERROR: {r['error']}")
            continue
        auc = "  -  " if r.get("auc") is None else f"{r['auc']:.4f}"
        lines.append(
            f"{r['name']:>10} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
            f"{r['recall']:>6.3f} {r['f1']:>6.3f} {auc}"
        )
    return to_json_safe({"results": results, "table": "\n".join(lines)})
