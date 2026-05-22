# LLE Curve Project (structured)

## Files

- config.py        : all hyperparams & paths
- utils.py         : seed, smiles canonicalization, fingerprints, scaler, t assignment
- data.py          : load/prepare excel, split, dataset
- model.py         : LLECurveNet
- metrics.py       : evaluation metrics, r2, loaders eval
- train.py         : train loop + curves
- predict.py       : pointwise prediction on df_raw_test
- viz.py           : parity + ternary plots (true tie-lines + pred tie-lines + metrics box)
- main.py          : entry point

## Run

1) Edit `EXCEL_PATH` in `config.py`
2) `python main.py`

Outputs will be in `OUT_DIR`.

这是我目前的工程代码，现在的代码将SMILES编码为分子指纹，我想通过RDKit将SMILES构建成图，然后用图神经网络进行特征提取，然后进行预测。
节点特征，包含但不限于：元素类型、原子度数、形式电荷、杂化态、是否芳香、总氢数、是否在环中、手性类别、原子质量、电负性、共价半径、范德华半径等物化性质。
边级特征，包括但不限于：键类型、是否共轭、是否在环中、立体化学、电负性差值、键长等。
还有其他和液液相分离的节点或者边特征，你也可以加进来

请你帮我修改目前的代码，修改后的代码必须能兼容之前的内容

目前的代码是对三个组分的SMILES分别构图，然后提取特征与温度拼接，最后预测，我想对三个组分构建混合物图，根据分子间相互作用构建分子之间的边，此外温度作为相互作用边的特征之一，然后提取每个分子的特征和混合物图的特征，再与温度拼接进行预测。

请帮我修改我目前的代码，但要保证能够兼容之前的代码，对于训练、预测、指标和可视化等代码，大体结构不要动，请给我完整可覆盖的代码

分子间相互作用包括以下但不限于以下内容：
1）范德华力是普遍存在于分子间的一种弱相互作用，其能量显著低于化学键。它主要由色散力、诱导力、取向力构成。
2）氢键是一种特殊且重要的分子间弱相互作用，它的本质兼具静电吸引和部分共价键的特性，其键能比范德华力强，但比化学键弱。氢键具有方向性和饱和性这两个重要特性并深刻影响物质性质。例如，水分子间广泛的氢键网络使其具有反常的高沸点、高比热容。
3）π-π堆积相互作用是芳香环体系间通过π电子云叠加产生的一种范德华力。其作用具有方向性，并易受芳香环上大取代基的位阻影响而减弱。
4）卤键是缺电子的卤素原子与富电子原子（如N、O）之间的静电作用。其形成源于卤素原子因与强电负性原子相连而电子云密度降低，呈现正电性。卤键键能虽较弱，但在生物体系与药物设计中具有重要作用。
RDKit的UFF/MMFF优化里，分子间主要考虑的相互作用有：范德华、经典、氢键、卤键、范德华与少量静电共同导致的环之间偏好接触或堆积。

把温度只加在“分子间边”，分子内边不加

增加更多 interaction 类型（离子-偶极 / 偶极-偶极 / 静电势估计等）】

 用 RDKit 生成 3D 构象后按几何距离严格建边（更物理）

这是我目前的代码，我现在想在此基础上加入官能团特征，让网络变成一种多尺度特征提取网络，最后将分子的官能团特征、单分子特征、混合物特征、温度特征进行拼接进行下游任务预测，下面的代码是别人工程里用于提取分子中官能团的代码，你看怎么能修改一下加入到我的代码中，让我的代码实现多尺度特征提取，请给我完整可覆盖的代码

目前的工程文件所提取的多尺度特征（三个组分的分子特征、每个组分的官能团特征、混合图图特征）和温度特征是通过concat然后送到MLP进行预测的，请你帮我修改为将每种特征划分为token，然后通过transformer进行融合然后预测

python src/fit_nrtl_params.py --excel_path "D:\GGNN\YXFL\data_update\update-LLE-all-with-smiles_min3.xlsx" --out_dir "nrtl_param" --alpha 0.3 --steps 3000 --lr 0.05 --g_max 8000.0




