# 温度编码与温度尾部稳健性

- 论文位置：补充信息 S3.3.2；表 S3-S4；图 S5
- 证据状态：代码和已有结果齐全
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/run_temperature_encoding_sensitivity.py`
- `scripts/plot_temperature_encoding_sensitivity.py`

## 运行入口

```powershell
python scripts/plot_temperature_encoding_sensitivity.py --help
```

## 说明

- 归档 seed 42、7、2024 的编码指标、距离分箱指标和汇总图；不复制中间训练权重。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
