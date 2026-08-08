# 热力学残差与化学势一致性审计

- 论文位置：补充信息 S3.7；表 S9-S10
- 证据状态：代码和已有结果齐全
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/analysis/evaluate_thermodynamic_consistency.py`
- `src/psmi/ge_models.py`
- `src/psmi/nrtl_flash.py`

## 运行入口

```powershell
python scripts/analysis/evaluate_thermodynamic_consistency.py --help
```

## 说明

- 公开逐预测残差、阈值敏感性和汇总 JSON，不包含任何答复草稿。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
