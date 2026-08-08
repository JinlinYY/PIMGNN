# 数据集说明

## `raw/`

保存来源工作簿和人工整理输入。预处理脚本不得覆盖这些文件。

## `processed/`

保存由 `scripts/data_preparation/` 生成的处理后数据。修正版主基准使用
`update-LLE-all-with-smiles.xlsx`，扩展微调使用
`LLE-literature-data-boosted.xlsx`。`baseline_comparison/` 是所有对比模型共用的
CSV 视图，必须由脚本重新生成，不能手工修改。

## `splits/`

保存固定的体系级划分清单。主基准为 612/75/78 个训练、验证、测试体系；扩展
数据为 575/72/72 个体系。任何增强都必须在读取清单并完成划分之后进行。

## `parameters/`

`corrected_v2/` 中的 NRTL 参数按用途隔离：

- `nrtl_params_train.json` 只含训练体系，可用于训练物理损失；
- `nrtl_params_all.json` 只用于检查点选择后的统一诊断；
- `nrtl_split_manifest.json` 记录数据哈希、拟合设置及精确体系范围。

训练入口会核验参数角色、数据哈希和体系集合，并在运行目录写入参数使用清单。
历史参数仅供来源追溯，不得支撑修正版论文指标。

## `external/`

保存不属于 PSMI 训练语料的第三方验证数据说明。使用外部数据前应单独确认许可、
来源和划分协议。
