# 应用案例流程

在仓库根目录执行预测、分析和绘图：

```powershell
python scripts/run_application_case.py `
  --excel "datasets/raw/应用案例-all-experiment.xlsx" `
  --ckpt "<经来源核验的检查点路径>" `
  --out_dir "experiments/09_application_cases/runs/reproduction"
```

只分析已有预测表：

```powershell
python scripts/run_application_case.py `
  --csv "<预测 CSV 路径>" `
  --out_dir "experiments/09_application_cases/runs/reproduction" `
  --analyze_only
```

应用案例必须记录检查点 SHA-256、输入数据 SHA-256、标量输入契约和生成命令。
不能用未经来源核验的历史目录名推断模型架构。
