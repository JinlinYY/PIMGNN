# 多尺度表示、跨分子交互与融合模块消融

- 论文位置：主文 3.1.2；表 2
- 证据状态：代码和已有指标齐全
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/train.py`
- `src/psmi/model.py`
- `configs/experiments`

## 运行入口

```powershell
python scripts/train.py --help
```

## 说明

- 保存现有多尺度融合和架构变体的最佳指标；不补跑历史消融训练。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
