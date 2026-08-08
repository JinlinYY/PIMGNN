# 数据划分实验

- `run_split_strategy_benchmark.py`：比较点随机、体系随机、组分家族和温度有序留出；
- `run_kfold_cv.py`：轮换体系互斥的测试折和验证折。

先用 `--dry-run` 生成并检查划分清单，确认体系零重叠后再训练。清单应记录联结线
阈值及组分 2/3 增强设置；增强只能应用于训练分区。

```powershell
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/data_splitting/run_split_strategy_benchmark.py --dry-run
E:\anaconda\envs\ggnn39\python.exe scripts/experiments/data_splitting/run_kfold_cv.py --dry-run
```
