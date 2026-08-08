# 外部数据转换脚本

这些命令整理 Abraham、CompSol 和 FreeSolv 的兼容转换流程，输出位于
`datasets/external/`，不会修改主 PSMI LLE 数据集。

```powershell
E:\anaconda\envs\ggnn39\python.exe scripts/data_preparation/public_release/convert_abraham.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/data_preparation/public_release/convert_compsol.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/data_preparation/public_release/download_freesolv.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/data_preparation/public_release/convert_freesolv_database.py --help
E:\anaconda\envs\ggnn39\python.exe scripts/data_preparation/public_release/build_freesolv_example.py --help
```

`build_freesolv_example.py` 只复现十行演示表，不能表述为完整 FreeSolv 基准。
