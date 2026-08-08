# 系带线密度阈值与相路径位置敏感性

- 论文位置：补充信息 S3.5；表 S6-S7；图 S6
- 证据状态：代码和已有结果齐全
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/run_tieline_threshold_sensitivity.py`
- `scripts/plot_tieline_threshold_sensitivity.py`
- `scripts/data_preparation/filter_sparse_systems.py`

## 运行入口

```powershell
python scripts/plot_tieline_threshold_sensitivity.py --help
```

## 说明

- 归档阈值 3-9 的总体指标、路径位置指标、置信区间、清单和汇总图。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
