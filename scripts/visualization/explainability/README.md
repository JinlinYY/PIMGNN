# 可解释性可视化

这些入口将 PSMI 重要性表转换为三组分分子图，共用解析和 RDKit 绘图逻辑位于
`_common.py`。

- `plot_node_importance.py`：原子或节点重要性；
- `plot_bond_importance.py`：键重要性及端点热图；
- `plot_functional_group_importance.py`：官能团连续热图；
- `plot_functional_group_importance_colorbar.py`：官能团纯色标记和色条。

示例：

```powershell
E:\anaconda\envs\ggnn39\python.exe scripts/visualization/explainability/plot_node_importance.py --limit 5
```

正式图必须记录数据、检查点、重要性算法、系统编号和生成命令。
