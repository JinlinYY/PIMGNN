# PSMI 三元液液相平衡 Web 应用

本应用提供 Vue 3 前端和 FastAPI 后端，用于输入三个组分的 SMILES、温度及曲线设置并生成 PSMI 三元 LLE 预测。应用由早期 Web 版本迁移而来，当前统一使用仓库中的 `src/psmi`，不再维护重复的模型源码。

## 目录结构

```text
PSMI-LLE-web/
├─ assets/explainability/   预计算可解释性摘要
├─ backend/                 FastAPI 与检查点适配器
├─ checkpoints/default/     默认历史 Web 检查点和官能团词表
├─ frontend/                Vue 3 + Vite 界面
├─ scripts/                 Windows 启动脚本
└─ tests/                   Web 冒烟测试
```

## 模型契约

后端在构造模型前读取检查点中的 `provenance.architecture`：

- 修正版检查点自动恢复标量维度、融合方式、混合物图节点布局和功能开关；
- 默认打包的历史 Web 检查点没有来源元数据，经权重审计确认使用两个标量 `[T, s]`、拼接融合和历史批处理布局；
- 因此默认检查点不使用压力。API 为兼容新的三标量 `[T, s, P]` 检查点仍接受压力字段，但响应中的 `pressure_used` 为 `false`，图标题也不显示压力；
- 加载带完整元数据的三标量修正版检查点后，压力会被标准化并作为正式输入，`pressure_used` 为 `true`。

这种处理避免把“接口接受压力”误写成“历史模型已经学习压力效应”。

## Python 环境

```powershell
conda activate ggnn39
python -m pip install -r Web\PSMI-LLE-web\requirements.txt
```

## 启动

在项目根目录打开两个 PowerShell 终端。

后端：

```powershell
Web\PSMI-LLE-web\scripts\run_backend.ps1
```

前端要求 Node.js `^20.19.0` 或 `>=22.12.0`：

```powershell
cd Web\PSMI-LLE-web\frontend
npm ci
cd ..\..\..
Web\PSMI-LLE-web\scripts\run_frontend.ps1
```

浏览器访问 `http://localhost:3000`；FastAPI 文档位于 `http://localhost:8000/docs`。

## 运行时变量

- `PSMI_WEB_DEVICE`：推理设备，默认 `cpu`；
- `PSMI_WEB_MODEL_PATH`：检查点路径；
- `PSMI_WEB_MODEL_DIR`：包含 `fg_corpus.json` 和可选 `last_model.pt` 的目录；
- `PSMI_WEB_EXPLAIN_DIR`：预计算可解释性目录；
- `PSMI_WEB_HOST`、`PSMI_WEB_PORT`：后端监听地址和端口；
- `VITE_API_BASE`：前端 API 基地址。

## 验证

```powershell
conda activate ggnn39
$env:PYTHONPATH='src;Web\PSMI-LLE-web'
python -m pytest tests\test_web_checkpoint_contract.py Web\PSMI-LLE-web\tests -q
cd Web\PSMI-LLE-web\frontend
npm ci
npm run build
```

来源记录中的旧模型名称仅用于说明历史迁移，不代表当前产品名称；面向用户的名称统一为 PSMI。
