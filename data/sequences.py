"""数据集划分与时序样本构建 (Task 1.5).

`build_sequences(full, labeled, look_backs, ...)` 把标签结果与完整因子日线转换为
多周期三维张量，并按时间序硬切分为 训练/验证/测试（base）或含 `valB` 的
增强切分（enhance）。

核心约定（对应Todo Task 1.5 / 需求§3.3~§3.5）：
    * 划分硬口径 训练/验证/测试 = 70%/15%/15%，**时序切分，禁 shuffle**，
      最新 15% 为测试集，比例放入配置可调。
    * 20/60/120 三个窗口**各自独立构建**，并**各自独立 Z-Score 标准化**；
      scaler **仅对训练集拟合参数**，验证/测试(及 valB)仅做 transform。
    * 禁止数组下标跨窗口对齐：不同 look_back 的窗口数可能不同（预热 warm-up
      导致起始差异），下游一律通过 trade_date 进行样本对齐（见 `dates` 与
      `sample_meta`，`date_to_split` 提供日期→划分映射）。
    * sample_weight：`enable_lowvol_weight=True` 时对 `low_volatility==1`
      （过去20日振幅<5%）的训练样本权重乘以 0.5；验证/测试权重恒为 1.0。
    * enhance 模式额外产出 `valB`（从验证带中再切出一段，仅用于集成权重，
      **禁止用于训练/早停**）；训练样本过少时自动降级回 base（`degraded_to_base`）。

标签在样本日 T 计算（用 T+1 开盘与窗口最后有效收盘），特征仅用到 T 为止，
两者通过 sample 的 trade_date 严格对应；构建窗口后标签贴在窗口末行（T日）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .labels import LabelResult

# 默认三窗口（需求§4.1.2）。
DEFAULT_LOOK_BACKS: tuple = (20, 60, 120)
# 硬口径划分比例，放入配置可调。
DEFAULT_SPLIT_RATIOS: dict = {"train": 0.70, "val": 0.15, "test": 0.15}
# enhance 模式下从"验证带"再切出的 valB 占比（相对全样本）。
DEFAULT_VALB_RATIO: float = 0.075
# enhance 降级阈值：valB 最少样本数，不足则自动降级回 base。
DEFAULT_VALB_MIN_SAMPLES: int = 10
VALID_SPLIT_MODES: tuple = ("base", "enhance")

# 构建特征列时的排除列（元数据 / 标签辅助 / OHLCV / 枚举）。
_RESERVED_EXCLUDED = frozenset({
    "trade_date", "date", "open", "high", "low", "close", "volume", "amount",
    "label", "return", "buy_price", "sell_price", "weight", "is_suspend",
    "float_shares",
})


@dataclass
class SequenceResult:
    """多周期序列构建结果。

    X / y / dates / weight 以 look_back 为键，均为**各自独立**的数组，长度可不同
    （预热差异），不得按下标跨窗口对齐。跨窗口对齐一律通过 `trade_date` 完成：
        * `dates[lb][i]` 是 X[lb][i] 对应窗口的 T 日；
        * `sample_meta` 是样本级(按 trade_date)元数据，含 label/split 归属；
        * `date_to_split[trade_date]` 给出该日期样本所属划分。
    """

    look_backs: List[int]
    split_mode: str                     # "base" | "enhance"
    feature_names: List[str]
    X: Dict[int, np.ndarray]            # LB -> [n, LB, F]
    y: Dict[int, np.ndarray]            # LB -> [n]  (0/1 标签)
    dates: Dict[int, pd.Index]          # LB -> [n] trade_date（窗口末行 T 日）
    weight: Dict[int, np.ndarray]       # LB -> [n]
    train_index: Dict[int, np.ndarray]  # LB -> 训练样本下标（按各自 X 数组）
    val_index: Dict[int, np.ndarray]    # LB -> 验证样本下标
    test_index: Dict[int, np.ndarray]   # LB -> 测试样本下标（最新15%）
    valb_index: Optional[Dict[int, np.ndarray]]  # None=base/降级
    scalers: Dict[int, object]          # LB -> StandardScaler（仅训练集拟合）
    sample_meta: pd.DataFrame           # 样本级 [trade_date,label,split] （升序）
    date_to_split: pd.Series            # trade_date -> split 名
    degraded_to_base: bool = False
    stats: dict = field(default_factory=dict)

    @property
    def has_valb(self) -> bool:
        return self.valb_index is not None


def _pick_feature_columns(full: pd.DataFrame, feature_columns) -> List[str]:
    if feature_columns is not None:
        cols = list(feature_columns)
        missing = [c for c in cols if c not in full.columns]
        if missing:
            raise ValueError(f"build_sequences: 缺少特征列 {missing}")
        return cols
    cands = [
        c for c in full.columns
        if c not in _RESERVED_EXCLUDED
        and pd.api.types.is_numeric_dtype(full[c])
    ]
    return cands


def _validate(full: pd.DataFrame, labeled: LabelResult) -> None:
    if not isinstance(full, pd.DataFrame) or full.empty:
        raise ValueError("build_sequences: full 必须是非空日线 DataFrame")
    if not isinstance(labeled, LabelResult):
        raise ValueError("build_sequences: labeled 必须为 LabelResult")
    if "trade_date" not in full.columns:
        raise ValueError("build_sequences: full 缺少 trade_date 列")
    if labeled.df.empty:
        raise ValueError("build_sequences: 标签结果无有效样本")
    # full 需为时间升序；排序由调用方保证，这里统一重排。
    full = full.sort_values("trade_date").reset_index(drop=True)
    if not full["trade_date"].is_monotonic_increasing:
        raise ValueError("build_sequences: full 需按 trade_date 升序")


def _assign_sample_splits(
    sample_df: pd.DataFrame,
    ratios: dict,
    enhance: bool,
    valb_ratio: float,
    valb_min_samples: int,
    embargo_days: int = 0,
) -> tuple[pd.DataFrame, pd.Series, bool]:
    """对样本级(按 trade_date 升序)分配划分，返回 (sample_meta, date_to_split,
    degraded_to_base)。"""
    dates = sample_df["trade_date"].to_numpy()
    n = len(dates)
    embargo_days = max(0, int(embargo_days))
    t = int(round(n * ratios["train"]))
    val_span = int(round(n * ratios["val"]))
    val_start = min(n, t + embargo_days)
    v_end = min(n, val_start + val_span)
    test_start = min(n, v_end + embargo_days)
    split_arr = np.full(n, "test", dtype=object)
    split_arr[:t] = "train"
    split_arr[t:val_start] = "embargo"
    split_arr[val_start:v_end] = "val"
    split_arr[v_end:test_start] = "embargo"

    degraded = False
    valb_split = None
    if enhance:
        # 从"验证带"内再切出 valB（仅用于集成权重）。
        vb_start = val_start
        vb_end = val_start + int(round(n * valb_ratio)) + 1
        valb_span = min(vb_end, v_end)
        if (valb_span - vb_start) >= valb_min_samples:
            valb_split = np.zeros(n, dtype=bool)
            valb_split[vb_start:valb_span] = True
            split_arr[valb_split] = "valB"
        else:
            degraded = True

    sample_meta = pd.DataFrame(
        {"trade_date": dates,
         "label": np.asarray(sample_df["label"], dtype=int),
         "split": split_arr},
        index=pd.RangeIndex(n),
    )
    date_to_split = pd.Series(
        split_arr, index=pd.Index(dates, name="trade_date"), dtype=object
    )
    date_to_split.index.name = "trade_date"
    return sample_meta, date_to_split, degraded


def build_sequences(
    full: pd.DataFrame,
    labeled: LabelResult,
    look_backs: tuple = DEFAULT_LOOK_BACKS,
    split_ratios: Optional[dict] = None,
    split_mode: str = "base",
    valb_ratio: float = DEFAULT_VALB_RATIO,
    valb_min_samples: int = DEFAULT_VALB_MIN_SAMPLES,
    enable_lowvol_weight: bool = True,
    feature_columns: Optional[List[str]] = None,
    embargo_days: int = 0,
) -> SequenceResult:
    """构建多周期时序样本并按时间序切分。

    Args:
        full: 完整升序日线（含因子列），用于提供每窗口至多 look_back-1 行的
            预热数据；须含 trade_date。
        labeled: `generate_labels` 的产物（其 `df` 为有效样本，行升序）。
        look_backs: 时间窗口集合，默认 (20, 60, 120)。
        split_ratios: {train,val,test} 硬划分比例，默认 70/15/15。
        split_mode: "base" 或 "enhance"。
        valb_ratio / valb_min_samples: enhance 模式 valB 切分占比与最小样本数。
        enable_lowvol_weight: 为 True 时训练样本 low_volatility==1 权重 ×0.5。
        feature_columns: 显式特征列；为 None 时自所有数值列中排除保留列得到。
        embargo_days: 训练/验证、验证/测试之间隔离的样本日期数量。

    Returns:
        SequenceResult（见其字段。三窗口各自独立标准化、独立张量）。
    """
    ratios = dict(DEFAULT_SPLIT_RATIOS)
    if split_ratios:
        ratios.update(split_ratios)
    s = sum(ratios.get(k, 0.0) for k in ("train", "val", "test"))
    if not np.isclose(s, 1.0):
        raise ValueError(f"build_sequences: train/val/test 比例之和须为 1，收到 {s:.3f}")
    if split_mode not in VALID_SPLIT_MODES:
        raise ValueError(f"split_mode 必须为 {VALID_SPLIT_MODES}，收到 {split_mode!r}")

    _validate(full, labeled)
    full = full.sort_values("trade_date").reset_index(drop=True)

    sample_df = labeled.df.sort_values("trade_date").reset_index(drop=True)
    feats = _pick_feature_columns(full, feature_columns)
    if len(feats) == 0:
        raise ValueError("build_sequences: 未找到任何数值特征列")

    # 样本级划分（按 trade_date，供所有窗口对齐，禁止跨窗口下标对齐）。
    sample_meta, date_to_split, degraded = _assign_sample_splits(
        sample_df, ratios, split_mode == "enhance", valb_ratio, valb_min_samples,
        embargo_days=embargo_days,
    )

    # full 中日期 -> 行位置，用于回溯窗口预热。
    full_dates = full["trade_date"].to_numpy()
    date_to_pos = {d: i for i, d in enumerate(full_dates)}
    feat_mat = full[feats].to_numpy(dtype=float)
    # 前向填充特征缺失（如 ret 首行 / ma5 前4行 NaN），避免其进入训练集
    # 标准化后产生 NaN 均值/方差；窗口首部等因素产生的 NaN 均按此补齐。
    if np.isnan(feat_mat).any():
        clean = pd.DataFrame(feat_mat, columns=feats).ffill().fillna(0.0)
        feat_mat = clean.to_numpy(dtype=float)

    # sample(标称 T 日) -> 依据
    sample_dates = sample_df["trade_date"].to_numpy()
    sample_labels = np.asarray(sample_df["label"], dtype=int)
    in_sample_pos = np.searchsorted(full_dates, sample_dates)

    X: Dict[int, np.ndarray] = {}
    y: Dict[int, np.ndarray] = {}
    dates_out: Dict[int, pd.Index] = {}
    weight: Dict[int, np.ndarray] = {}
    train_index: Dict[int, np.ndarray] = {}
    val_index: Dict[int, np.ndarray] = {}
    test_index: Dict[int, np.ndarray] = {}
    valb_index: Dict[int, np.ndarray] = {}
    scalers: Dict[int, object] = {}

    from sklearn.preprocessing import StandardScaler

    lb_list = sorted(set(int(lb) for lb in look_backs))
    for lb in lb_list:
        # 对每个 sample 构造长度为 lb 的窗口 [pos-lb+1 .. pos]。
        keep = []
        wins = []
        for k in range(len(sample_dates)):
            pos = in_sample_pos[k]
            if pos < lb - 1:
                continue  # 预热不足：该样本在 lb 窗口下无法构建。
            start = pos - lb + 1
            wins.append(feat_mat[start:pos + 1])
            keep.append(k)
        if not wins:
            # 该窗口完全没有可构建样本。
            X[lb] = np.empty((0, lb, len(feats)), dtype=float)
            y[lb] = np.empty((0,), dtype=int)
            dates_out[lb] = pd.Index([], name="trade_date")
            weight[lb] = np.empty((0,), dtype=float)
            train_index[lb] = np.empty((0,), dtype=int)
            val_index[lb] = np.empty((0,), dtype=int)
            test_index[lb] = np.empty((0,), dtype=int)
            scalers[lb] = StandardScaler()
            continue

        arr = np.stack(wins, axis=0).astype(float)   # [n, lb, F]
        keep = np.asarray(keep, dtype=int)
        labels = sample_labels[keep]                 # [n]
        dts = sample_dates[keep]                     # [n]
        wts = np.ones(len(keep), dtype=float)

        # 训练样本低波动减权。
        if enable_lowvol_weight:
            lowvol = np.asarray(sample_df["low_volatility"].to_numpy()[keep] == 1,
                                dtype=bool) if "low_volatility" in sample_df.columns \
                else np.zeros(len(keep), dtype=bool)
            # 仅对 train 划分样本施加 0.5 权重。
            tmask = date_to_split.reindex(pd.Index(dts)).to_numpy() == "train"
            wts[tmask & lowvol] = 0.5

        # 划分下标（按该窗口自身的索引）。
        split_of_date = np.asarray(
            [date_to_split[d] for d in dts], dtype=object
        )
        tidx = np.where(split_of_date == "train")[0]
        vidx = np.where(split_of_date == "val")[0]
        teidx = np.where(split_of_date == "test")[0]
        bidx = np.where(split_of_date == "valB")[0]

        # 独立 Z-Score：scaled 值按 [n, F] 扁平拟合训练集，再变换全部。
        sc = StandardScaler()
        flat = arr.reshape(-1, lb * len(feats))
        sc.fit(flat[tidx])
        arr_scaled = sc.transform(flat).reshape(arr.shape)

        X[lb] = arr_scaled
        y[lb] = labels
        dates_out[lb] = pd.Index(dts, name="trade_date")
        weight[lb] = wts
        train_index[lb] = tidx
        val_index[lb] = vidx
        test_index[lb] = teidx
        valb_index[lb] = bidx
        scalers[lb] = sc

    stats = {
        "split_mode": split_mode,
        "ratios": dict(ratios),
        "enable_lowvol_weight": bool(enable_lowvol_weight),
        "n_samples": int(len(sample_meta)),
        "per_window": {lb: int(len(y[lb])) for lb in lb_list},
        "per_split_train": {lb: int(len(train_index[lb])) for lb in lb_list},
        "per_split_val": {lb: int(len(val_index[lb])) for lb in lb_list},
        "per_split_test": {lb: int(len(test_index[lb])) for lb in lb_list},
        "degraded_to_base": degraded,
    }

    return SequenceResult(
        look_backs=lb_list,
        split_mode=split_mode,
        feature_names=feats,
        X=X,
        y=y,
        dates=dates_out,
        weight=weight,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        valb_index=None if (degraded or split_mode == "base") else valb_index,
        scalers=scalers,
        sample_meta=sample_meta,
        date_to_split=date_to_split,
        degraded_to_base=degraded,
        stats=stats,
    )
