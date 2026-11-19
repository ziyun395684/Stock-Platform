"""Build the current strategy research decision matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _summary(root: Path, relative: str, cost: float) -> pd.Series:
    frame = pd.read_csv(root / relative)
    rows = frame[frame["cost_rate"].eq(cost)]
    if rows.empty:
        raise ValueError(f"缺少成本率 {cost}: {relative}")
    return rows.iloc[0]


def build_report(root: Path) -> str:
    candidate = _summary(
        root,
        "output/research/factor_baseline/time_oos_250_cost005/summary.csv",
        0.005,
    )
    external = _summary(
        root,
        "output/research/factor_baseline/external_etf_oos/summary.csv",
        0.005,
    )
    filter_summary = pd.read_csv(
        root / "output/research/factor_baseline/market_regime_filter/summary.csv"
    )
    fixed_no_filter = filter_summary.query(
        "pool == 'fixed' and filter == 'no_filter'"
    ).iloc[0]
    fixed_up_only = filter_summary.query(
        "pool == 'fixed' and filter == 'up_only'"
    ).iloc[0]
    double_fixed = pd.read_csv(
        root / "output/research/factor_baseline/double_oos/fixed_etf_report.csv"
    )
    double_external = pd.read_csv(
        root / "output/research/factor_baseline/double_oos/external_etf_report.csv"
    )

    lines = [
        "# StockRNN 策略研究决策矩阵",
        "",
        "> 生成日期：2026-09-22。本报告用于研究和辅助交易决策，不连接券商，也不构成实盘建议。",
        "",
        "## 决策矩阵",
        "",
        "| 方向 | 当前证据 | 决策 | 原因 |",
        "|---|---|---|---|",
        f"| Factor-common + Top-2 + 20 日 | 固定 ETF 主窗口成本 0.005 平均净收益 {candidate.mean_net_return:.2%}，正收益 {int(candidate.positive_windows)}/{int(candidate.n_windows)} | **候选主线** | 成本后仍为正，但市场阶段和外部 ETF 泛化证据不足 |",
        f"| Factor-common 外部 ETF | 成本 0.005 平均净收益 {external.mean_net_return:.2%}，正收益 {int(external.positive_windows)}/{int(external.n_windows)} | **研究分支** | 上涨阶段有效，下跌/震荡阶段为负 |",
        f"| 市场状态过滤 | 固定池不过滤 {fixed_no_filter.mean_net_return:.2%}，上涨过滤 {fixed_up_only.mean_net_return:.2%} | **研究分支** | 可降回撤但交易数量明显减少，未形成稳定增量 |",
        f"| 双重样本外 | 固定池 {double_fixed.total_return_net.mean():.2%}，外部池 {double_external.total_return_net.mean():.2%} | **风险门禁未通过** | 两类 ETF 池均出现明显阶段依赖 |",
        "| Transformer residual | 长测试段多 seed 为负，平均最大回撤较高 | **不推荐** | 排序泛化不稳定，未超过传统因子基线 |",
        "| Factor + Transformer 混合 | Transformer 权重增加后收益和回撤恶化 | **不推荐** | 未形成可重复的增量价值 |",
        "| 原复合排序策略 | 长测试段明显弱于 Factor-common | **仅作 Baseline** | 保留用于历史对照，不作为新主线 |",
        "| Top-1/Top-3/Top-5 | 仅作为敏感性结果 | **仅作对照** | Top-2 风险收益更均衡，不根据单阶段切换 |",
        "",
        "## 当前执行建议",
        "",
        "1. 辅助交易研究候选固定为 Factor-common + Top-2 + 20 日持有。",
        "2. 默认不启用 Transformer 混合和市场状态过滤。",
        "3. 使用固定 ETF 池进行时间样本外复核，保留外部 ETF 结果作为风险提示。",
        "4. 输出人工复核计划，不接入券商和自动下单系统。",
        "5. 当前不冻结为最终自动化策略，继续要求跨阶段和跨标的证据。",
        "",
        "## 仍未通过的门禁",
        "",
        "- 高波动阶段：125 日补充窗口成本 0.005 下仅 1/2 窗口盈利。",
        "- 外部 ETF 跨阶段泛化：上涨 4/4，下跌 0/2，震荡 0/2。",
        "- 双重样本外：固定 ETF 池 2/8 窗口盈利，外部 ETF 池 4/8 窗口盈利，不能视为稳定泛化。",
        "- 结论不能由单个历史阶段或单个测试窗口决定。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="output/research/reports/strategy_decision_matrix.md",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(root), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
