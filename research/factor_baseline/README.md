# Factor Baseline

用于提取多套旧策略的共性因子，并构建不依赖神经网络的 Factor-Only 基线。

首批候选因子：

```text
roc_5
roc_10
roc_20
roc_60
trend_score_24
ma_energy_41
volume_ratio
relative_return_to_benchmark
```

所有因子必须只使用信号日及以前的数据。因子权重只能在训练集或验证集确定，测试集只用于最终报告。
