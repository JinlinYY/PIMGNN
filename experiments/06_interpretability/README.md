# 模型可解释性与特征重要性可视化

- 论文位置：主文 3.2；图 2b-2e
- 证据状态：代码和已有结果齐全
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/explain.py`
- `scripts/visualization/explainability`

## 运行入口

```powershell
python scripts/explain.py --help
```

## 说明

- 公开特征重要性表、混合物节点和边重要性表，以及代表性可视化。
- 历史可解释性输出中的体系编号与论文图注需人工核对，仓库不擅自改写结果。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
