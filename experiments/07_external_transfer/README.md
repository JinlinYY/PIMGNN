# 二元溶解度外部验证与扩展 LLE 微调

- 论文位置：主文 3.3；图 2f；表 S2
- 证据状态：代码、扩展结果和权重齐全，外部指标表待确认
- 整理原则：直接归档工程中已有文件，本次未重新训练模型。

## 代码入口

- `scripts/experiments/transfer_learning/public_release`
- `src/psmi_legacy_public`

## 运行入口

```powershell
python scripts/experiments/transfer_learning/public_release/evaluate_predictions.py --help
```

## 说明

- 公开兼容脚本和历史迁移权重位于 models/06_transfer_learning 与 models/07_external_validation。
- expanded_lle corrected_v2 的三个随机种子结果和最佳权重随实验目录归档。
- 历史外部二元溶解度归档未包含可确认的原始数据表和指标表，因此表 S2 只能提供代码与权重证据，不能从当前仓库重新核验数值。

## 已归档内容

本目录中的 `results`、`figures` 或 `data` 来自当前工程已有输出。
