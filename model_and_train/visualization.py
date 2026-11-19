"""Task 4.1 可视化图表实现.

对应 Todo Task4.1 / 需求 §5.4：

    图表清单（自动保存 PNG）：
        1. 训练 loss/acc 迭代曲线        plot_training_history
        2. 预测概率 vs 真实标签           plot_prob_vs_label
        3. ROC 曲线 + AUC                plot_roc_curve
        4. Precision-Recall 曲线          plot_pr_curve
        5. 混淆矩阵热力图                 plot_confusion_matrix
        6. 集成权重分布条形图             plot_ensemble_weights
        7. 回测净值曲线                   plot_backtest_nav（取 BacktestReport.nav_curve）
        8. 子模型预测相关性热力图         plot_correlation_heatmap

中文乱码处理：统一 `setup_chinese_font()`，选用中文字体（SimHei/
Microsoft YaHei 等），关闭 unicode_minus 负号异常；全部使用
matplotlib "Agg" 无界面后端，保证无显示环境下正常出图。
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: 中文字体候选（Windows 优先 SimHei / Microsoft YaHei）。
_CHINESE_FONTS = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]


def setup_chinese_font() -> None:
    """配置 matplotlib 中文字体，避免标签乱码（幂等）。"""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams["font.sans-serif"] = _CHINESE_FONTS
    plt.rcParams["axes.unicode_minus"] = False
    # 中文字体缺失时 matplotlib 会告警，不影响出图（仅乱码）。
    plt.rcParams["font.family"] = "sans-serif"


def _save(fig, save_path: str) -> str:
    """保存图并关闭，返回路径。"""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if not save_path.lower().endswith(".png"):
        save_path = save_path + ".png"
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    from matplotlib import pyplot as plt

    plt.close(fig)
    logger.info("图表已保存: %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# 1. 训练 loss/acc 曲线
# ---------------------------------------------------------------------------
def plot_training_history(
    history: Dict[str, Sequence[float]],
    save_path: str,
    title: str = "训练曲线",
) -> str:
    """训练 loss / 指标迭代曲线（左 loss，右 acc+f1 等）。"""
    setup_chinese_font()
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history.get("loss", [])) + 1)

    # 左图：loss
    ax = axes[0]
    if history.get("loss"):
        ax.plot(epochs, history["loss"], label="loss")
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], label="val_loss")
    ax.set_title(f"{title} - Loss")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    # 右图：accuracy / f1 等指标
    ax = axes[1]
    for key in ("accuracy", "f1", "precision", "recall"):
        if history.get(key):
            ax.plot(epochs, history[key], label=key)
    if history.get("val_accuracy"):
        ax.plot(epochs, history["val_accuracy"], label="val_accuracy")
    if history.get("val_f1"):
        ax.plot(epochs, history["val_f1"], label="val_f1")
    ax.set_title(f"{title} - 指标")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 2. 预测概率 vs 真实标签
# ---------------------------------------------------------------------------
def plot_prob_vs_label(
    dates: Sequence,
    y_true: Sequence[int],
    prob: Sequence[float],
    save_path: str,
    title: str = "预测概率 vs 真实标签",
) -> str:
    """按时间轴对齐绘制概率曲线与标签散点。"""
    setup_chinese_font()
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(prob))
    ax.plot(x, prob, label="预测概率", color="tab:blue", lw=1.2)
    y = np.asarray(y_true)
    pos_idx = x[y == 1]
    ax.scatter(pos_idx, np.ones(len(pos_idx)) * 1.02,
               marker="|", color="tab:red", s=30, label="正例标签")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="阈值 0.5")
    ax.set_title(title)
    ax.set_xlabel("样本序号（按日期升序）")
    ax.set_ylabel("概率")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 3. ROC 曲线 + AUC
# ---------------------------------------------------------------------------
def plot_roc_curve(
    y_true: Sequence[int],
    prob: Sequence[float],
    save_path: str,
    title: str = "ROC 曲线",
) -> str:
    """ROC 曲线 + AUC（sklearn 口径）。"""
    from sklearn.metrics import auc, roc_curve

    setup_chinese_font()
    from matplotlib import pyplot as plt

    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(prob))
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_title(title)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 4. PR 曲线
# ---------------------------------------------------------------------------
def plot_pr_curve(
    y_true: Sequence[int],
    prob: Sequence[float],
    save_path: str,
    title: str = "Precision-Recall 曲线",
) -> str:
    """PR 曲线 + Average Precision（sklearn 口径）。"""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    setup_chinese_font()
    from matplotlib import pyplot as plt

    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    precision, recall, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 5. 混淆矩阵
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    save_path: str,
    title: str = "混淆矩阵",
) -> str:
    """混淆矩阵热力图（seaborn）。"""
    from sklearn.metrics import confusion_matrix

    setup_chinese_font()
    from matplotlib import pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["负类(0)", "正类(1)"],
                yticklabels=["负类(0)", "正类(1)"], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 6. 集成权重分布
# ---------------------------------------------------------------------------
def plot_ensemble_weights(
    weights: Dict[int, float],
    save_path: str,
    title: str = "集成权重分布",
) -> str:
    """各窗口子模型集成权重条形图。"""
    setup_chinese_font()
    from matplotlib import pyplot as plt

    lbs = sorted(weights.keys())
    vals = [weights[lb] for lb in lbs]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([str(lb) for lb in lbs], vals, color="tab:blue")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("look_back 窗口")
    ax.set_ylabel("集成权重")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 7. 回测净值曲线
# ---------------------------------------------------------------------------
def plot_backtest_nav(
    report,
    save_path: str,
    title: str = "策略回测净值曲线",
) -> str:
    """回测净值曲线（取自 BacktestReport.nav_curve，需求§5.4）。"""
    setup_chinese_font()
    from matplotlib import pyplot as plt

    nav = np.asarray(report.nav_curve, dtype="float64")
    fig, ax = plt.subplots(figsize=(12, 5))
    if len(nav):
        ax.plot(nav, lw=1.6, color="tab:green", label="净值(扣费后)")
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title(
            f"{title}\n累计收益(gross)={report.total_return_gross:.2%} | "
            f"net={report.total_return_net:.2%} | 胜率={report.win_rate}"
        )
    else:
        ax.text(0.5, 0.5, "无成交", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    ax.set_xlabel("成交序号")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 8. 相关性热力图
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(
    correlation_matrix: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "子模型预测相关性",
) -> str:
    """子模型预测 Spearman 相关矩阵热力图。"""
    import pandas as pd

    setup_chinese_font()
    from matplotlib import pyplot as plt
    import seaborn as sns

    if not correlation_matrix:
        raise ValueError("相关性矩阵为空，无法绘图")
    df = pd.DataFrame(correlation_matrix).astype(float)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, save_path)
