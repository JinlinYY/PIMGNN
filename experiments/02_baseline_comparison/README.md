# 通用机器学习与图神经网络对比实验

- 论文位置：主文 3.1.1；表 1
- 证据状态：代码和已有结果齐全
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/experiments/baselines/run_classical_multi_seed.py`
- `scripts/experiments/baselines/run_cignn.py`
- `scripts/experiments/baselines/run_cgib.py`
- `scripts/experiments/baselines/run_glam.py`
- `src/psmi_baselines`

## 运行入口

```powershell
python scripts/experiments/baselines/run_classical_multi_seed.py --help
```

## 说明

- 归档论文对比表使用的历史五随机种子汇总。其划分协议与 corrected_v2 主基准不同，不能混合解读。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
