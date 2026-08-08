# 过量吉布斯能模型敏感性

- 论文位置：补充信息 S3.6；表 S8
- 证据状态：代码和已有结果齐全
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/experiments/ge_model_sensitivity.py`
- `src/psmi/ge_models.py`

## 运行入口

```powershell
python scripts/experiments/ge_model_sensitivity.py --help
```

## 说明

- 归档 data-only、NRTL、Margules 和 van Laar 的逐种子指标、配对比较、参数与划分清单；省略大体积重复权重。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
