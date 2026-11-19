# Legacy Strategy Replay

用于复测 `strage/策略与数据/` 中的旧策略，验证策略描述能否在当前 StockRNN 数据、信号和组合回测口径下复现。

当前入口仍位于 `scripts/`，本目录先保存分类说明和后续复测专用脚本。复测结果写入 `output/research/legacy_replay/`。

复测必须记录：策略文件、ETF 池、数据起止日期、成本、Top-K、退出规则、CAGR、最大回撤、Sharpe、Calmar 和换手率。
