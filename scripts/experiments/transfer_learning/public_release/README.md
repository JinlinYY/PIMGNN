# 历史公开迁移实验

这些入口通过 `psmi_legacy_public` 复现 Abraham、CompSol 和 BigSolDB 的历史兼容
流程。它们不代表当前 `corrected_v2` 主基准。

```powershell
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/transfer_learning/public_release/train_abraham.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/transfer_learning/public_release/train_compsol.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/transfer_learning/public_release/finetune_bigsoldb_degenerate.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/transfer_learning/public_release/finetune_bigsoldb_frozen_head.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/transfer_learning/public_release/predict_bigsoldb.py --help
```

原始来源未包含完整外部数据表，并采用确定性行顺序 80/20 划分。因此其指标必须
明确标注为历史迁移实验，不能与体系互斥的主论文指标直接混用。
