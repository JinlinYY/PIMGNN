# 对比实验命令

先在 `ggnn39` 环境和仓库根目录生成统一数据表：

```powershell
E:\anaconda\envs\ggnn39\python.exe scripts/data_preparation/prepare_baseline_comparison_data.py
```

随后通过相应入口运行通用模型、CGIB、CIGNN、GLAM、MMGNN 或 SolvBERT：

```powershell
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/baselines/run_classical.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/baselines/run_classical_multi_seed.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/baselines/run_cgib.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/baselines/run_cignn.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/baselines/run_glam.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/baselines/run_mmgnn.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/baselines/run_solvbert.py --help
```

## 固定划分协议

生成脚本导出 `corrected_v2` 的 612/75/78 体系划分。MMGNN 和 SolvBERT 读取
三个独立分区文件；CGIB、CIGNN 和 GLAM 读取 `total.csv` 中同源的 `split`
标签并检查体系零重叠；通用模型直接使用同一清单。训练期间只用验证集选模，测试
集在最佳权重确定后评估。

修正前由随机行划分得到的历史结果不能自动升级，也不能混入修正版对比表。
多种子入口默认将每个种子放在同一版本化输出目录下，并写出包含数据与划分哈希的
`protocol.json`；只有所有预期模型的指标文件齐全时才会跳过已有种子。
