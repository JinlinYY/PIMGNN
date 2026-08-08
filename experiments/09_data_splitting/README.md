# 体系级划分、交叉验证与划分策略

- 论文位置：补充信息 S3.4；表 S5
- 证据状态：已有代码和划分清单，完整指标待确认
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/experiments/data_splitting/run_kfold_cv.py`
- `scripts/experiments/data_splitting/run_split_strategy_benchmark.py`

## 运行入口

```powershell
python scripts/experiments/data_splitting/run_kfold_cv.py --help
```

## 说明

- 当前仅确认固定划分、K 折和策略对比清单；未找到可验证的表 S5 完整指标文件，因此不补写数值。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
