# PSMI：三元液液相平衡预测

本仓库公开 PSMI 的数据处理、图神经网络模型、热力学约束、论文实验、已有结果、最佳权重、结果绘图和 Web 应用。公开模型名称统一为 **PSMI**。

## 项目结构

```text
PSMI/
├─ configs/                 数据、模型与实验配置
├─ datasets/                处理后数据、固定划分和热力学参数
├─ src/psmi/                PSMI 模型与训练实现
├─ src/psmi_baselines/      对比模型实现
├─ scripts/                 训练、评估、分析和绘图入口
├─ experiments/             按论文顺序整理的代码入口与已有结果
├─ models/                  主模型、迁移模型和兼容权重
├─ Web/PSMI-LLE-web/        FastAPI + Vue Web 应用
├─ tests/                   单元测试和回归测试
└─ docs/                    模型、结果与使用说明
```

## 论文实验

论文主文和补充信息中的实验统一列在 [实验索引](experiments/README.md)。每个实验子目录均说明：

- 对应的论文章节、表格或图；
- 实际代码入口和运行命令；
- 已归档的指标、预测表、结果图或最佳权重；
- 当前证据状态和需要人工确认的历史缺口。

所有归档结果均来自工程中已有文件，本次代码整理没有重新训练模型。主基准、扩展 LLE、多模型对比、架构消融、物理正则化、误差分析、可解释性、温度编码、系带线敏感性、过量吉布斯能模型敏感性、热力学审计、体系泛化、工业案例和效率实验均提供相应代码及现有证据。局部温度扰动和数据划分部分缺少可确认的完整历史结果，目录中已如实标注。

## 环境

推荐使用 `ggnn39`：

```powershell
conda env create -f environment.yml
conda activate ggnn39
```

也可以在现有环境中安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 主模型运行入口

仅使用已发布权重核对结果（不会训练或修改权重）：

```powershell
python scripts/reproduce_current_weights.py --device cuda
python scripts/reproduce_current_weights.py `
  --registry configs/reproduction/historical_paper_weight_registry.json `
  --output-root results/paper_reproduction/historical_weight_inference `
  --device cuda
python scripts/analysis/build_paper_reproduction_bundle.py
```

整理后的论文报告表、逐点预测、结果图和协议说明见
[`results/paper_reproduction/`](results/paper_reproduction/README.md)。论文历史协议和
`corrected_v2` 修正协议分别归档，不能混用指标。

下列入口用于确实需要重新训练的情形：

监督训练：

```powershell
python scripts/train.py --config configs/experiments/main_benchmark_stage1.yaml
```

物理约束微调：

```powershell
python scripts/train.py --config configs/experiments/main_benchmark_stage2.yaml
```

扩展 LLE 数据微调：

```powershell
python scripts/train.py --config configs/experiments/expanded_lle_finetune.yaml
```

固定划分清单位于 `datasets/splits/`。推荐主科学版本为 `corrected_v2`；历史权重与修正版权重不可混用。

## 测试

```powershell
$env:PYTHONPATH='src'
python -m pytest -q
```

公开版本应在 `ggnn39` 中运行完整测试，并通过公开目录安全审计；实际通过数量以当前测试输出为准。

## Web 应用

```powershell
Web/PSMI-LLE-web/scripts/run_backend.ps1
Web/PSMI-LLE-web/scripts/run_frontend.ps1
```

详细说明见 [Web 使用文档](Web/PSMI-LLE-web/README.md)。

## 科学边界

当前主配置通过统一的 NRTL 过量 Gibbs 自由能形式计算活度系数，并约束两相化学势一致性，但没有启用独立的 Gibbs–Duhem 残差损失。相关科学表述和代码边界见 [科学模型契约](docs/architecture/scientific_model_contract_cn.md)。
