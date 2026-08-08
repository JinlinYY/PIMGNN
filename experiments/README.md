# 论文实验索引

本目录按照论文主文和补充信息的实验顺序组织。代码入口统一保留在 scripts、src、configs 和 Web 中；每个实验目录提供对应入口、已有结果及证据状态。构建过程未重新训练模型。

| 目录 | 论文位置 | 证据状态 |
| --- | --- | --- |
| [数据集构建、筛选与分布分析](00_dataset_construction/) | 主文 2.1；补充信息 S5；表 S15-S16；图 S8 | 代码和已有结果齐全 |
| [PSMI 主基准与多随机种子评估](01_main_benchmark/) | 主文 3.1；表 1；图 2a | 代码、已有结果和最佳权重齐全 |
| [通用机器学习与图神经网络对比实验](02_baseline_comparison/) | 主文 3.1.1；表 1 | 代码和已有结果齐全 |
| [多尺度表示、跨分子交互与融合模块消融](03_architecture_ablation/) | 主文 3.1.2；表 2 | 代码和已有指标齐全 |
| [物理约束正则化对比](04_physics_regularization/) | 主文 2.3；主文 3.1.2；表 3 | 代码和已有结果齐全 |
| [预测误差分布与偏差分析](05_prediction_error_analysis/) | 补充信息 S3.1；图 S1-S3 | 代码和历史归档结果齐全 |
| [模型可解释性与特征重要性可视化](06_interpretability/) | 主文 3.2；图 2b-2e | 代码和已有结果齐全 |
| [二元溶解度外部验证与扩展 LLE 微调](07_external_transfer/) | 主文 3.3；图 2f；表 S2 | 代码、扩展结果和权重齐全，外部指标表待确认 |
| [局部温度扰动与相路径敏感性](08_temperature_robustness/01_local_perturbation/) | 补充信息 S3.3.1；图 S4 | 已有代码，未找到可确认的保存结果 |
| [温度编码与温度尾部稳健性](08_temperature_robustness/02_encoding_and_tail/) | 补充信息 S3.3.2；表 S3-S4；图 S5 | 代码和已有结果齐全 |
| [体系级划分、交叉验证与划分策略](09_data_splitting/) | 补充信息 S3.4；表 S5 | 已有代码和划分清单，完整指标待确认 |
| [系带线密度阈值与相路径位置敏感性](10_tieline_sensitivity/) | 补充信息 S3.5；表 S6-S7；图 S6 | 代码和已有结果齐全 |
| [过量吉布斯能模型敏感性](11_ge_model_sensitivity/) | 补充信息 S3.6；表 S8 | 代码和已有结果齐全 |
| [热力学残差与化学势一致性审计](12_thermodynamic_audit/) | 补充信息 S3.7；表 S9-S10 | 代码和已有结果齐全 |
| [体系级泛化与相图类别统计](13_system_generalization/) | 补充信息 S3.8；表 S11-S14；表 S17；补充信息 S6 | 代码和已有结果齐全 |
| [工业应用预测与分析工作流](14_industrial_cases/00_application_workflow/) | 主文 3.4 | 代码和已有结果齐全 |
| [环丁砜芳烃抽提案例](14_industrial_cases/01_aromatic_extraction/) | 主文 3.4.1；图 3c | 代码、数据和已有图齐全 |
| [DEM 回收案例](14_industrial_cases/02_dem_recovery/) | 主文 3.4.2；图 3d | 代码、数据和已有图齐全 |
| [推理效率与运行时间基准](15_efficiency/) | 补充信息 S3.9 | 代码和已有结果齐全 |
| [PSMI-LLE Web 应用](16_web_application/) | 补充信息 S4；图 S7 | 代码和默认权重齐全 |
