# 物理约束正则化对比

- 论文位置：主文 2.3；主文 3.1.2；表 3
- 证据状态：代码和已有结果齐全
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/train.py`
- `src/psmi/loss.py`
- `src/psmi/ge_models.py`
- `src/psmi/nrtl_flash.py`

## 运行入口

```powershell
python scripts/train.py --help
```

## 说明

- 监督阶段与物理约束阶段的逐种子指标来自主基准 corrected_v2 结果。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
