# 运行脚本

所有正式入口均从仓库根目录运行，并自动加载本地 `src/` 包。

- `train.py`：按分层 YAML 配置训练、验证和最终测试；
- `fit_nrtl.py`：分别拟合训练损失参数和训练后诊断参数；
- `explain.py`：模型可解释性分析；
- `evaluate_case.py`、`run_application_case.py`：单案例评估与应用分析；
- `data_preparation/`：数据清洗、固定划分及对比表生成；
- `experiments/`：对比、划分稳健性和迁移实验入口；
- `analysis/`：冻结结果的后处理分析；
- `visualization/`：数据、结果和可解释性绘图；
- `maintenance/`：公开发布构建与安全审计。

## 现有权重的仅推理复现

```powershell
E:\anaconda\envs\ggnn39\python.exe scripts\reproduce_current_weights.py --device cuda
E:\anaconda\envs\ggnn39\python.exe scripts\reproduce_current_weights.py `
  --registry configs\reproduction\historical_paper_weight_registry.json `
  --output-root results\paper_reproduction\historical_weight_inference `
  --device cuda
E:\anaconda\envs\ggnn39\python.exe scripts\analysis\build_paper_reproduction_bundle.py
```

`evaluate_checkpoint.py` 会严格加载已有检查点并输出逐点预测、指标、奇偶图和审计 JSON；
`reproduce_current_weights.py` 按注册表批量调用它。两者均不包含优化器步骤。

主基准 NRTL 参数的推荐重建命令为：

```powershell
E:\anaconda\envs\ggnn39\python.exe scripts\fit_nrtl.py `
  --excel_path datasets\processed\update-LLE-all-with-smiles.xlsx `
  --out_dir datasets\parameters\corrected_v2 --scope both `
  --split-strategy manifest `
  --split-manifest datasets\splits\main_benchmark_corrected_v2.json `
  --min-points 6 --steps 3000 --device cuda
```

新实验数字应优先引用带固定划分、数据哈希、配置和检查点来源元数据的运行。论文历史结果若缺少完整来源元数据，只能作为已归档证据，并须与修正协议分开说明。
