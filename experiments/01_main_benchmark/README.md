# PSMI 主基准与多随机种子评估

- 论文位置：主文 3.1；表 1；图 2a
- 证据状态：代码、已有结果和最佳权重齐全
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/train.py`
- `src/psmi/predict.py`
- `scripts/experiments/run_corrected_multiseed.py`

## 运行入口

```powershell
python scripts/experiments/run_corrected_multiseed.py --help
```

## 说明

- 保存的是 corrected_v2 固定体系级划分下 seed 42、43、44 的监督阶段和物理约束阶段结果。
- 直接使用已有结果和最佳权重，未重新训练。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
