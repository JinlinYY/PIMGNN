# web_backend/utils/smiles_utils.py
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np
from typing import List, Tuple, Optional
import config as C
import importlib.util
import os

_PROJECT_UTILS_MODULE = None

def _load_project_utils_module():
    """显式加载src/utils.py为独立模块，避免与web_backend.utils冲突。"""
    global _PROJECT_UTILS_MODULE
    if _PROJECT_UTILS_MODULE is not None:
        return _PROJECT_UTILS_MODULE

    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "utils.py")
    )
    spec = importlib.util.spec_from_file_location("project_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError("Failed to load src/utils.py")
    loader.exec_module(module)
    _PROJECT_UTILS_MODULE = module
    return module

def validate_smiles(smiles: str) -> bool:
    """验证SMILES格式"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def smiles_to_fingerprint(smiles: str, bits: int = 2048) -> np.ndarray:
    """将SMILES转换为Morgan指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
    return np.array(fp)

def smiles_to_graph(smiles: str, add_hs: bool = False, add_3d: bool = False, use_gasteiger: bool = True) -> dict:
    """将SMILES转换为图结构（使用原项目的完整实现）"""
    # 显式加载原项目utils模块，避免与web_backend.utils冲突
    project_utils = _load_project_utils_module()
    original_smiles_to_graph = project_utils.smiles_to_graph

    # 调用原项目的smiles_to_graph函数
    graph_dict = original_smiles_to_graph(
        smiles=smiles,
        add_hs=add_hs,
        add_3d=add_3d,
        use_gasteiger=use_gasteiger
    )

    # 转换为PyTorch张量格式
    import torch
    result = {
        'x': torch.tensor(graph_dict['x'], dtype=torch.float),
        'edge_index': torch.tensor(graph_dict['edge_index'], dtype=torch.long),
        'edge_attr': torch.tensor(graph_dict['edge_attr'], dtype=torch.float),
        'g': torch.tensor(graph_dict['g'], dtype=torch.float).unsqueeze(0),  # 添加batch维度
        'batch': torch.zeros(graph_dict['x'].shape[0], dtype=torch.long)
    }

    return result

def prepare_model_input(smiles_list: List[str], temperature: float, scaler) -> dict:
    """准备模型输入"""
    if C.USE_GRAPH:
        graphs = [smiles_to_graph(s) for s in smiles_list]
        scalars = torch.tensor([temperature, 0.5], dtype=torch.float)  # 简化，t=0.5
        return {'g1': graphs[0], 'g2': graphs[1], 'g3': graphs[2], 'scalars': scalars}
    else:
        fps = [smiles_to_fingerprint(s) for s in smiles_list]
        fp_concat = np.concatenate(fps)
        t_norm = scaler.transform([[temperature]])[0][0]
        input_vec = np.concatenate([fp_concat, [t_norm, 0.5]])
        return torch.tensor(input_vec, dtype=torch.float)