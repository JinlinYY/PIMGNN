# 配置文件说明

配置按数据、模型、训练阶段和完整实验四层组织。运行时可以重复使用
`--config`，后面的配置覆盖前面的同名参数；完整实验配置也可以通过
`include` 引用其他配置。

```powershell
conda activate ggnn39
python scripts/train.py --config configs/experiments/main_benchmark_stage1.yaml
```

`psmi_corrected_v2.yaml` 使用与批处理图索引一致的样本优先节点顺序。历史权重
只能在显式选择 `legacy_component_major` 时复现旧行为，不得把旧结果标记成修正
模型结果。

主要基准配置使用温度和相路径两个标量。扩展数据配置额外使用标准化压力，并且
必须在论文方法中明确说明这一输入差异。

