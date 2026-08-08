# 数据集构建、筛选与分布分析

- 论文位置：主文 2.1；补充信息 S5；表 S15-S16；图 S8
- 证据状态：代码和已有结果齐全
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/data_preparation/build_canonical_split.py`
- `scripts/data_preparation/build_expanded_split.py`
- `scripts/data_preparation/filter_sparse_systems.py`
- `scripts/analysis/analyze_dataset_distribution.py`
- `scripts/analysis/export_temperature_range_audit.py`

## 运行入口

```powershell
python scripts/analysis/analyze_dataset_distribution.py --help
python scripts/data_preparation/build_canonical_split.py --help
```

## 说明

- 公开数据、固定划分清单、组分统计和数据分布图均随仓库提供。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
