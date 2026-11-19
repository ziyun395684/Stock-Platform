# StockRNN Research Workspace

本目录用于隔离策略复测、共性因子研究和模型对照实验。现有运行入口、数据目录、模型代码和测试目录保持不变。

## 目录约定

```text
research/
├─ legacy_replay/       旧策略复测入口和复测说明
├─ factor_baseline/     共性因子库与 Factor-Only 基线
├─ model_comparison/    Transformer、混合模型和残差模型对照
├─ diagnostics/         窗口、种子、过拟合和阶段性诊断
└─ reports/             研究报告、结果索引和结论
```

## 文件归属规则

- `scripts/` 保留现有可执行脚本，确保历史命令和结果可复现。
- 新的研究脚本优先放到对应的 `research/` 子目录；稳定后再决定是否提升为正式 `scripts/` 入口。
- `model_and_train/` 只放可复用实现，不放一次性实验脚本。
- `tests/` 只放自动化测试，不放手工实验结果。
- 实验输出统一写入 `output/research/<category>/`，不要直接散落在 `output/` 根目录。
- `strage/策略与数据/` 是外部旧策略参考区，不在本次整理中移动或修改。

## 当前脚本映射

### 旧策略复测

| 现有脚本 | 研究分类 | 结果目录 |
| --- | --- | --- |
| `scripts/replay_legacy_long_term.py` | `legacy_replay` | `output/research/legacy_replay/` |
| `scripts/replay_legacy_ma_energy.py` | `legacy_replay` | `output/research/legacy_replay/` |
| `scripts/compare_execution_rules.py` | `legacy_replay` | `output/research/legacy_replay/` |

### 因子和模型研究

| 现有脚本 | 研究分类 |
| --- | --- |
| `scripts/run_feature_ablation.py` | `factor_baseline` |
| `scripts/run_target_diagnostics.py` | `factor_baseline` |
| `scripts/compare_three_models.py` | `model_comparison` |
| `scripts/run_ridge_rolling.py` | `model_comparison` |
| `scripts/run_ranking_diagnostics.py` | `diagnostics` |

## 研究主线

```text
旧策略复测
    -> 共性因子抽取
    -> Factor-Only 基线
    -> Transformer-Only 对照
    -> Factor + Transformer
    -> 残差学习
    -> 滚动样本外验收
```

每个新实验需要记录：数据区间、ETF 池、持仓规则、成本、训练窗口、验证规则、随机种子和输出文件位置。

## 辅助交易 CLI

`scripts/auxiliary_signal_cli.py` 只生成本地执行计划和人工复核清单，不连接券商、不提交订单：

```powershell
C:\Users\Windows\anaconda3\envs\p311\python.exe scripts/auxiliary_signal_cli.py `
  --predictions predictions.csv `
  --prices prices.csv `
  --as-of-date 20260919 `
  --output-dir output/auxiliary_signal `
  --top-k 2 `
  --hold-days 10 `
  --model-version model_v1 `
  --training-window 1000 `
  --seed 42 `
  --cost-rate 0.003 `
  --plan-version v1
```

输出包括 `plan.json`、`signals.csv` 和 `manual_checklist.csv`。计划中的 T+1 开盘可成交性会根据价格面板检查，异常标的会保留拒单原因。
CLI 会先校验预测标的和交易日都存在于价格面板中，避免生成跨面板或无法复核的计划。
已有输出目录中的 `plan.json` 默认不会被覆盖；确认需要重新生成时，命令行显式添加 `--overwrite`。

### Factor-common 研究候选模式

在不提供 Transformer 预测文件时，可以直接从价格面板生成 Factor-common
研究候选信号。该模式只生成辅助交易计划，不替换默认 Transformer 流程：

```powershell
C:\Users\Windows\anaconda3\envs\p311\python.exe scripts/auxiliary_signal_cli.py `
  --strategy-mode factor_common `
  --prices prices_with_close_volume.csv `
  --as-of-date 20260919 `
  --output-dir output/auxiliary_signal_factor `
  --top-k 2 `
  --hold-days 20 `
  --factor-mode common `
  --cost-rate 0.003 `
  --universe-symbols 518880.SH,513100.SH,513500.SH
```

Factor-common 模式的价格文件至少需要 `trade_date`、`symbol`、`open`、
`close` 和 `volume` 字段，可选 `--benchmark-path` 提供基准价格。计划元数据
会记录 `strategy_mode`、`factor_mode`、信号日、执行日、计划退出日、Top-K、持有期、成本率和固定 ETF 池。
信号日必须同时满足以下条件，才能生成可执行的人工复核计划：

- 信号日后存在 T+1 交易日；
- 信号日后存在完整的 `hold_days` 个交易日；
- 选中标的在 T+1 开盘可成交。

如果信号日接近数据尾部，计划器不会把退出日强行截到最后一个数据日，而是明确提示数据最后交易日、建议最晚信号日和“没有完整持有窗口”。固定持有 20 个交易日时，不应直接使用价格面板的最后一个交易日作为信号日。

如果原始数据仍按 `quotes/<symbol>.csv` 分文件保存，可以直接使用研究 runner，
它会自动生成价格面板、基准快照和人工复核计划：

```powershell
C:\Users\Windows\anaconda3\envs\p311\python.exe research/factor_baseline/run_auxiliary_candidate.py `
  --data-dir quotes `
  --output-dir output/auxiliary_signal_factor `
  --symbols 518880.SH,513100.SH,513500.SH,162719.SZ `
  --benchmark-symbol 510300.SH `
  --as-of-date 20250929 `
  --top-k 2 `
  --hold-days 20 `
  --factor-mode common `
  --cost-rate 0.003 `
  --initial-capital 100000 `
  --plan-version factor-common-v1
```

runner 输出的 `prices_panel.csv`、`benchmark.csv`、`plan.json`、`signals.csv`、
`manual_checklist.csv` 和 `plan_summary.json/.md` 会保存在同一目录，便于复盘和重复生成。

人工复核时优先查看 `plan_summary.md`，再逐笔检查 `manual_checklist.csv`；成交后使用
`scripts.record_auxiliary_fills` 回写，不修改原始信号快照。
成交回写会自动刷新同目录的 `plan_summary.json/.md`，摘要中的订单状态会更新为
`filled`、`partially_filled`、`fill_rejected` 或仍待处理状态。

持仓退出后可回写实际退出信息：

```powershell
C:\Users\Windows\anaconda3\envs\p311\python.exe -m scripts.record_auxiliary_exits `
  --plan output/auxiliary_signal_factor/plan.json `
  --exits exits.csv `
  --checklist-output output/auxiliary_signal_factor/manual_checklist.csv
```

退出文件至少包含 `symbol`、`actual_exit_date`、`actual_exit_price`，可选
`exit_reason` 和 `exit_notes`。回写后订单状态变为 `closed`，并同步刷新计划摘要。

退出回写完成后，也会生成 `auxiliary_review.json/.md`，统计已完成订单的毛收益、
扣成本净收益、胜率和持有自然日。该报告只统计同时存在成交价和实际退出价的订单。
如果计划元数据包含 `initial_capital`，报告还会计算已实现资金基数、已实现净利润和
已实现资金收益率；未成交订单不计入这些已实现指标。

人工成交回写：

```powershell
C:\Users\Windows\anaconda3\envs\p311\python.exe -m scripts.record_auxiliary_fills `
  --plan output/auxiliary_signal/plan.json `
  --fills fills.csv `
  --checklist-output output/auxiliary_signal/manual_checklist.csv
```

成交文件支持 CSV 或 JSON。CSV 至少包含 `symbol`，部分成交可提供
`filled`、`filled_quantity` 和 `fill_price`；同一订单的多次部分成交会累计保存，
并记录到 `fill_history`。计划外标的、重复成交和跨计划日期回写会被拒绝。该流程只
维护本地计划与人工复核结果，不接入券商、不自动下单。
