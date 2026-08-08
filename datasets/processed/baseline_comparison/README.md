# 对比模型共享数据

本目录由 `scripts/data_preparation/prepare_baseline_comparison_data.py` 生成：

- `train.csv`：6,092 条记录、612 个训练体系；
- `validation.csv`：788 条记录、75 个验证体系；
- `test.csv`：803 条记录、78 个测试体系；
- `total.csv`：以上原始记录及其 `split` 标签；
- `split_manifest.csv`：体系划分、源数据哈希和固定清单哈希。

所有分区在组分排列增强前按 `system_id` 确定，并严格互斥。文件同时保留当前列名
和导入对比代码需要的兼容列名。请重新运行生成脚本，不要手工编辑这些派生文件。
