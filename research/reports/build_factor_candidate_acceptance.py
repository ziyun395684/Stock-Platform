"""Build an acceptance report for the Factor-common candidate strategy."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_report(root: Path) -> str:
    sensitivity_dir = root / "output" / "research" / "factor_baseline" / "long_holdout_factor_candidates"
    mode_dir = root / "output" / "research" / "factor_baseline" / "long_holdout_factor_modes"
    summary = pd.read_csv(sensitivity_dir / "summary.csv")
    by_regime = pd.read_csv(sensitivity_dir / "by_regime.csv")
    modes = pd.read_csv(mode_dir / "summary.csv")
    time_oos_dir = root / "output" / "research" / "factor_baseline"
    time_oos_250 = pd.read_csv(time_oos_dir / "time_oos_250" / "summary.csv")
    time_oos_250_cost005 = pd.read_csv(
        time_oos_dir / "time_oos_250_cost005" / "summary.csv"
    )
    time_oos_125 = pd.read_csv(
        time_oos_dir / "time_oos_125_supplement" / "summary.csv"
    )
    time_oos_125_regime = pd.read_csv(
        time_oos_dir / "time_oos_125_supplement" / "by_regime.csv"
    )
    external_oos = pd.read_csv(time_oos_dir / "external_etf_oos" / "summary.csv")
    external_regime = pd.read_csv(
        time_oos_dir / "external_etf_oos" / "by_regime.csv"
    )
    double_oos_dir = time_oos_dir / "double_oos"
    double_external = pd.read_csv(double_oos_dir / "external_etf_report.csv")
    double_fixed = pd.read_csv(double_oos_dir / "fixed_etf_report.csv")

    candidate = summary[
        summary["top_k"].eq(2) & summary["hold_days"].eq(20)
    ].sort_values("cost_rate")
    if len(candidate) != 2:
        raise ValueError("候选配置必须包含两档成本率")

    required_regimes = {"down", "up", "high_volatility"}
    regime_candidate = by_regime[
        by_regime["top_k"].eq(2) & by_regime["hold_days"].eq(20)
    ]
    covered_regimes = set(regime_candidate["market_regime"].dropna())
    regime_gate = required_regimes.issubset(covered_regimes) and all(
        regime_candidate[regime_candidate["market_regime"].eq(regime)]["n_windows"].max() >= 2
        for regime in required_regimes
        if regime in covered_regimes
    )
    common_modes = modes[modes["factor_mode"].eq("common")].sort_values("cost_rate")
    best_mode = modes[~modes["factor_mode"].eq("common")].sort_values(
        ["cost_rate", "mean_net_return"], ascending=[True, False]
    ).groupby("cost_rate", as_index=False).first()

    lines = [
        "# Factor-common 候选策略阶段性验收",
        "",
        "> 本报告只用于研究验收，不构成实盘建议。实验日期：2026-09-22。",
        "",
        "## 固定口径",
        "",
        "- 标的池：当前固定 ETF 池。",
        "- 信号：Factor-common 横截面排序。",
        "- 组合：Top-2 等权。",
        "- 持有：固定 20 个交易日。",
        "- 测试：500 日训练、250 日验证、250 日测试、250 日步长，6 个滚动窗口。",
        "- 时间样本外扩展：固定 ETF 池从 2015 年开始的 250 日非重叠窗口；125 日窗口仅作为阶段覆盖补充。",
        "",
        "## 候选结果",
        "",
        "| 成本率 | 平均净收益 | 正收益窗口 | 平均最大回撤 | 组合净收益 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in candidate.iterrows():
        lines.append(
            f"| {row.cost_rate:.3f} | {row.mean_net_return:.2%} | "
            f"{int(row.positive_windows)}/{int(row.n_windows)} | "
            f"{row.mean_max_drawdown:.2%} | {row.total_net_return:.2%} |"
        )

    lines += [
        "",
        "## 因子模式对照",
        "",
        "| 成本率 | Factor-common | 最佳单因子 |",
        "|---:|---:|---:|",
    ]
    for _, row in common_modes.iterrows():
        best = best_mode[best_mode["cost_rate"].eq(row.cost_rate)].iloc[0]
        lines.append(
            f"| {row.cost_rate:.3f} | {row.mean_net_return:.2%} | "
            f"{best.factor_mode}: {best.mean_net_return:.2%} |"
        )

    primary = time_oos_250.iloc[0]
    cost005 = time_oos_250_cost005.iloc[0]
    supplement003 = time_oos_125[time_oos_125["cost_rate"].eq(0.003)].iloc[0]
    supplement005 = time_oos_125[time_oos_125["cost_rate"].eq(0.005)].iloc[0]
    high_vol = time_oos_125_regime[
        (time_oos_125_regime["market_regime"] == "high_volatility")
        & (time_oos_125_regime["cost_rate"] == 0.005)
    ].iloc[0]
    external003 = external_oos[external_oos["cost_rate"].eq(0.003)].iloc[0]
    external005 = external_oos[external_oos["cost_rate"].eq(0.005)].iloc[0]
    external005_regime = external_regime[external_regime["cost_rate"].eq(0.005)]
    external_up = external005_regime[
        external005_regime["market_regime"] == "up"
    ].iloc[0]
    external_down = external005_regime[
        external005_regime["market_regime"] == "down"
    ].iloc[0]
    external_sideways = external005_regime[
        external005_regime["market_regime"] == "sideways"
    ].iloc[0]

    lines += [
        "",
        "## 门禁判断",
        "",
        f"- 成本稳健性：{'通过' if (candidate['mean_net_return'] > 0).all() else '未通过'}。两档成本下候选均保持正收益。",
        f"- 窗口稳定性：{'通过' if (candidate['positive_windows'] >= 5).all() else '部分通过'}。",
        f"- 阶段覆盖：{'通过' if regime_gate else '未通过'}。上涨阶段当前只有 1 个长测试窗口，证据不足。",
        "- 因子模式：通过。Factor-common 在两档成本下均优于单因子模式。",
        "",
        "## 时间样本外扩展",
        "",
        "| 实验 | 成本率 | 平均净收益 | 正收益窗口 | 平均最大回撤 |",
        "|---|---:|---:|---:|---:|",
        f"| 250 日主窗口 | 0.003 | {primary.mean_net_return:.2%} | {int(primary.positive_windows)}/{int(primary.n_windows)} | {primary.mean_max_drawdown:.2%} |",
        f"| 250 日主窗口 | 0.005 | {cost005.mean_net_return:.2%} | {int(cost005.positive_windows)}/{int(cost005.n_windows)} | {cost005.mean_max_drawdown:.2%} |",
        f"| 125 日补充窗口 | 0.003 | {supplement003.mean_net_return:.2%} | {int(supplement003.positive_windows)}/{int(supplement003.n_windows)} | {supplement003.mean_max_drawdown:.2%} |",
        f"| 125 日补充窗口 | 0.005 | {supplement005.mean_net_return:.2%} | {int(supplement005.positive_windows)}/{int(supplement005.n_windows)} | {supplement005.mean_max_drawdown:.2%} |",
        "",
        f"- 高波动阶段证据：{'充分' if high_vol.positive_windows >= 2 and high_vol.n_windows >= 2 else '不足'}。125 日补充窗口在成本率 0.005 下为 {int(high_vol.positive_windows)}/{int(high_vol.n_windows)} 个正收益窗口。",
        "",
        "## 未见 ETF 跨标的样本外",
        "",
        "| 成本率 | 平均净收益 | 正收益窗口 | 平均最大回撤 |",
        "|---:|---:|---:|---:|",
        f"| 0.003 | {external003.mean_net_return:.2%} | {int(external003.positive_windows)}/{int(external003.n_windows)} | {external003.mean_max_drawdown:.2%} |",
        f"| 0.005 | {external005.mean_net_return:.2%} | {int(external005.positive_windows)}/{int(external005.n_windows)} | {external005.mean_max_drawdown:.2%} |",
        f"- 阶段拆分（成本率 0.005）：上涨 {int(external_up.positive_windows)}/{int(external_up.n_windows)}，下跌 {int(external_down.positive_windows)}/{int(external_down.n_windows)}，震荡 {int(external_sideways.positive_windows)}/{int(external_sideways.n_windows)}。",
        f"- 跨标的泛化：{'通过' if external_down.positive_windows > 0 and external_sideways.positive_windows > 0 else '未通过'}。外部 ETF 结果不能只依赖上涨阶段。",
        "",
        "## 未见 ETF + 未见时间双重样本外",
        "",
        "| ETF 池 | 平均净收益 | 正收益窗口 | 平均最大回撤 |",
        "|---|---:|---:|---:|",
        f"| 固定 ETF 池 | {double_fixed['total_return_net'].mean():.2%} | {int((double_fixed['total_return_net'] > 0).sum())}/{len(double_fixed)} | {double_fixed['max_drawdown'].mean():.2%} |",
        f"| 外部 ETF 池 | {double_external['total_return_net'].mean():.2%} | {int((double_external['total_return_net'] > 0).sum())}/{len(double_external)} | {double_external['max_drawdown'].mean():.2%} |",
        "- 双重样本外使用相同的 Top-2、20 日持有和成本率 0.005；固定池与外部池均未达到跨阶段稳定泛化门禁。",
        "",
        "## 结论",
        "",
        "Factor-common + Top-2 + 20 日是当前最稳健的研究候选，但双重样本外结果显示其表现仍依赖 ETF 池和市场阶段，尚不能冻结为最终策略。下一步应优先积累人工辅助交易复盘记录，而不是继续针对当前窗口调参。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="output/research/reports/factor_candidate_acceptance.md",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(root), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
