"""Task 3.1 多周期集成模块 EnsembleModel.

对应 Todo Task3.1 / 需求 §4.2.4 / §8.4：

    * 支持 `baseline` / `enhanced` 双模式：
        - baseline：集成权重在**验证集(val)**上拟合，w_i = F1_i / Σ F1；
        - enhanced：集成权重在**valB**上拟合（valB 仅用于集成权重，
          禁止训练/早停，见 data.sequences 契约）。
    * `fit_weights()`：F1‑NaN 的窗口权重置 0 后归一化；原始 F1 值（含
      NaN）转 None 用于 JSON 报告。
    * `predict()`：返回 `tuple(集成概率数组, date_dict)`；**禁止数组下标
      跨窗口对齐** —— 各窗口样本日期不同（预热差异），一律按 trade_date
      Join 对齐，date_dict 保存各窗口样本日期列表与对齐后的公共日期。
    * `correlation_report()`：子模型预测 Spearman 相关矩阵；若任意一对
      相关系数 > 0.8（需求§8.4），输出告警并给出降级方案（精简为单一
      最优子模型 或 切换 Stacking）。

集成可用性（Task2.4 验收衔接）：子模型缺失/禁用的窗口直接跳过，
仅用可用窗口加权；全部不可用时报错。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: 高相关阈值（需求§8.4：Spearman > 0.8 视为高度相关）。
HIGH_CORRELATION_THRESHOLD: float = 0.8

VALID_MODES: tuple = ("baseline", "enhanced")


@dataclass
class EnsembleWeights:
    """集成权重与各窗口 F1（JSON 安全：NaN -> None）。"""

    mode: str
    weights: Dict[int, float] = field(default_factory=dict)   # lb -> 归一化权重
    f1_values: Dict[int, Optional[float]] = field(default_factory=dict)  # lb -> F1 或 None(NaN)
    degraded: bool = False

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "weights": {str(k): float(v) for k, v in self.weights.items()},
            "f1_values": {str(k): (None if v is None else float(v))
                          for k, v in self.f1_values.items()},
            "degraded": self.degraded,
        }


class EnsembleModel:
    """多周期子模型加权集成（Task 3.1）。

    与子模型解耦：本类只消费各窗口的预测概率数组 + 日期列表 + 标签，
    不直接依赖 keras 模型（子模型由 `LSTMAttentionModel` 产出概率）。
    """

    def __init__(
        self,
        mode: str = "baseline",
        infer_threshold: float = 0.5,
        high_correlation_threshold: float = HIGH_CORRELATION_THRESHOLD,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode 必须为 {VALID_MODES}，收到 {mode!r}")
        self.mode = mode
        self.infer_threshold = float(infer_threshold)
        self.high_correlation_threshold = float(high_correlation_threshold)
        self.weights: EnsembleWeights = EnsembleWeights(mode=mode)
        self._submodel_prob: Dict[int, np.ndarray] = {}
        self._submodel_dates: Dict[int, np.ndarray] = {}
        self._submodel_y: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 数据装载（prob/date/y 均按 lb 组织，跨窗口通过 trade_date 对齐）
    # ------------------------------------------------------------------
    def set_window_data(
        self,
        prob: Dict[int, np.ndarray],
        dates: Dict[int, Sequence],
        y: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """装载各窗口在**权重拟合集**（base=val / enhanced=valB）上的预测。

        Args:
            prob: lb -> 上涨概率数组 [n_lb]。
            dates: lb -> 与 prob 对齐的 trade_date 列表 [n_lb]。
            y: lb -> 标签 [n_lb]（fit_weights 需要；predict-only 可省略）。
        """
        for lb, p in prob.items():
            p = np.asarray(p, dtype="float64")
            d = np.asarray(list(dates[lb]))
            if p.shape[0] != d.shape[0]:
                raise ValueError(
                    f"窗口 {lb}: prob 长度 {p.shape[0]} 与 dates 长度 {d.shape[0]} 不一致"
                )
            self._submodel_prob[lb] = p
            self._submodel_dates[lb] = d
            if y is not None and lb in y:
                yy = np.asarray(y[lb])
                if yy.shape[0] != p.shape[0]:
                    raise ValueError(
                        f"窗口 {lb}: y 长度 {yy.shape[0]} 与 prob 长度 {p.shape[0]} 不一致"
                    )
                self._submodel_y[lb] = yy

    # ------------------------------------------------------------------
    # 权重拟合
    # ------------------------------------------------------------------
    @staticmethod
    def _window_f1(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Optional[float]:
        """单窗口 F1；无正例/无样本时返回 None（视为 NaN，权重置 0）。"""
        if len(y) == 0:
            return None
        pos = int(np.sum(np.asarray(y) == 1))
        if pos == 0:
            return None
        pred = (np.asarray(p) >= threshold).astype(int)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            return 0.0
        f1 = 2 * prec * rec / (prec + rec)
        return None if not np.isfinite(f1) else float(f1)

    def fit_weights(self, threshold: float = 0.5) -> EnsembleWeights:
        """按各窗口 F1 拟合集成权重。

        规则：
            * base 模式：权重集应为 val；enhanced 模式：应为 valB
              （由调用方保证传入的 set_window_data 数据来自哪个划分）。
            * F1 为 None(NaN) 的窗口权重置 0；剩余窗口按
              w_i = F1_i / Σ F1 归一化（需求§4.2.4）。
            * 全部 F1 无效时告警并退化为等权（degraded=True）。
        """
        if not self._submodel_prob:
            raise RuntimeError("fit_weights 前必须先 set_window_data")
        f1_values: Dict[int, Optional[float]] = {}
        weights: Dict[int, float] = {}
        for lb in sorted(self._submodel_prob.keys()):
            if lb in self._submodel_y:
                f1 = self._window_f1(self._submodel_y[lb], self._submodel_prob[lb], threshold)
            else:
                f1 = None
                logger.warning("窗口 %d 缺少标签，无法计算 F1，权重置 0", lb)
            f1_values[lb] = f1

        finite = {lb: v for lb, v in f1_values.items() if v is not None}
        total = sum(finite.values())
        degraded = False
        if total > 0:
            for lb in f1_values:
                v = f1_values[lb]
                weights[lb] = 0.0 if v is None else float(v / total)
        else:
            degraded = True
            logger.warning(
                "fit_weights: 全部窗口 F1 无效(NaN/无正例)，退化为等权集成"
            )
            n = len(f1_values)
            for lb in f1_values:
                weights[lb] = 1.0 / n if n else 0.0

        self.weights = EnsembleWeights(
            mode=self.mode,
            weights=weights,
            f1_values=f1_values,
            degraded=degraded,
        )
        return self.weights

    # ------------------------------------------------------------------
    # 集成预测（trade_date Join，禁止下标跨窗口对齐）
    # ------------------------------------------------------------------
    def predict(
        self,
        prob: Optional[Dict[int, np.ndarray]] = None,
        dates: Optional[Dict[int, Sequence]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """集成预测：返回 (集成概率数组, date_dict)。

        date_dict:
            {
              "per_window": {lb: [trade_date, ...]},   # 各窗口样本日期
              "aligned": [trade_date, ...],            # 公共日期（对齐后）
              "per_window_aligned_prob": {lb: [...]},  # 对齐后的各窗口概率
            }

        对齐规则：取所有可用窗口日期的**交集**（公共日期），在公共日期上
        加权求和；缺失窗口自动跳过。
        """
        if prob is not None and dates is not None:
            self.set_window_data(prob, dates)
        if not self._submodel_prob:
            raise RuntimeError("predict 前必须先 set_window_data 或传入 prob/dates")
        if not self.weights.weights:
            self.fit_weights()

        # 公共日期 = 各窗口日期集合的交集（保持升序）。
        date_sets = [set(self._submodel_dates[lb]) for lb in self._submodel_prob]
        common = sorted(set.intersection(*date_sets)) if date_sets else []
        if not common:
            raise ValueError("各窗口无公共交易日，无法集成预测（禁止下标对齐）")

        date_index = {d: i for i, d in enumerate(common)}
        aligned_prob: Dict[int, np.ndarray] = {}
        for lb in self._submodel_prob:
            dates_lb = self._submodel_dates[lb]
            prob_lb = self._submodel_prob[lb]
            pos = np.searchsorted(dates_lb, common)
            valid = pos < len(dates_lb)
            pos = np.where(valid, pos, 0)
            vals = prob_lb[pos]
            vals = np.where(valid, vals, np.nan)
            aligned_prob[lb] = vals

        w = self.weights.weights
        total_w = sum(w.get(lb, 0.0) for lb in aligned_prob)
        if total_w <= 0:
            raise RuntimeError("集成权重全为 0，无法预测（无可用子模型）")

        ens = np.zeros(len(common), dtype="float64")
        for lb, vals in aligned_prob.items():
            ens = ens + w.get(lb, 0.0) * np.nan_to_num(vals, nan=0.0)
        ens = ens / total_w

        date_dict = {
            "per_window": {lb: list(self._submodel_dates[lb]) for lb in self._submodel_prob},
            "aligned": common,
            "per_window_aligned_prob": {lb: v.tolist() for lb, v in aligned_prob.items()},
        }
        return ens, date_dict

    def predict_binary(self, ens: np.ndarray) -> np.ndarray:
        """按 infer_threshold 把集成概率转 0/1。"""
        return (np.asarray(ens) >= self.infer_threshold).astype(int)

    # ------------------------------------------------------------------
    # 相关性报告与降级决策（需求§8.4）
    # ------------------------------------------------------------------
    def correlation_report(
        self,
        prob: Optional[Dict[int, np.ndarray]] = None,
        dates: Optional[Dict[int, Sequence]] = None,
    ) -> dict:
        """输出子模型预测 Spearman 相关矩阵 + 高相关告警 + 降级方案。

        对齐方式与 predict 相同（trade_date 交集）。
        """
        if prob is not None and dates is not None:
            self.set_window_data(prob, dates)
        lbs = sorted(self._submodel_prob.keys())
        if len(lbs) < 2:
            return {
                "correlation_matrix": {},
                "max_correlation": None,
                "warnings": ["子模型数量 < 2，无法计算相关性"],
                "degradation_suggestion": None,
            }

        common_dates = set.intersection(*(set(self._submodel_dates[lb]) for lb in lbs))
        common = sorted(common_dates)
        if len(common) < 2:
            return {
                "correlation_matrix": {},
                "max_correlation": None,
                "warnings": ["公共交易日不足 2 个，无法计算相关性"],
                "degradation_suggestion": None,
            }

        from scipy.stats import spearmanr

        mat: Dict[str, Dict[str, float]] = {str(lb): {} for lb in lbs}
        pairs_high: List[Tuple[int, int, float]] = []
        for i, la in enumerate(lbs):
            pa = self._prob_at_dates(la, common)
            for j, lb in enumerate(lbs):
                if j <= i:
                    continue
                pb = self._prob_at_dates(lb, common)
                rho, _ = spearmanr(pa, pb)
                rho = float(rho) if np.isfinite(rho) else 0.0
                mat[str(la)][str(lb)] = rho
                mat[str(lb)][str(la)] = rho
                if abs(rho) > self.high_correlation_threshold:
                    pairs_high.append((la, lb, rho))
            mat[str(la)][str(la)] = 1.0

        max_corr = None
        flat = [v for row in mat.values() for v in row.values()]
        if flat:
            max_corr = round(max(abs(v) for v in flat), 4)

        warnings: List[str] = []
        suggestion = None
        if pairs_high:
            detail = "；".join(
                f"窗口 {a}×{b}: ρ={r:.2f}" for a, b, r in pairs_high
            )
            warnings.append(
                f"高相关告警：存在 Spearman 相关系数 > {self.high_correlation_threshold} "
                f"的子模型对（{detail}），集成增益可能有限（需求§8.4）"
            )
            best_lb = max(
                (lb for lb in lbs if self.weights.f1_values.get(lb) is not None),
                key=lambda lb: self.weights.f1_values.get(lb) or 0.0,
                default=None,
            )
            suggestion = (
                f"降级方案：①精简为单一最优子模型（窗口 {best_lb}，按 F1 最大）；"
                "②切换 Stacking 策略"
                if best_lb is not None
                else "降级方案：①精简为单一最优子模型（无有效 F1，按权重最大）；②切换 Stacking 策略"
            )
        return {
            "correlation_matrix": mat,
            "max_correlation": max_corr,
            "warnings": warnings,
            "degradation_suggestion": suggestion,
        }

    def _prob_at_dates(self, lb: int, dates: Sequence) -> np.ndarray:
        d = self._submodel_dates[lb]
        p = self._submodel_prob[lb]
        pos = np.searchsorted(d, dates)
        return p[pos]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """集成权重与配置摘要（JSON 安全）。"""
        return {
            "mode": self.mode,
            "infer_threshold": self.infer_threshold,
            "weights": self.weights.to_dict(),
        }
