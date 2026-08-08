# 分析脚本

本目录保存读取冻结数据、预测表或检查点的后处理工具，训练入口仍位于
`scripts/` 根目录。

- `analyze_dataset_distribution.py`：统计数据覆盖、过滤、组分、温度和联结线密度；
- `run_sensitivity_analysis.py`：计算温度有限差分敏感性和相路径扫描；
- `classify_phase_diagram_generalization.py`：生成留出体系的相图与联结线分类。

分析输出必须与运行配置、输入文件哈希和检查点哈希绑定。公开实验说明只使用科学
问题和论文小节命名，不包含内部工作记录。
