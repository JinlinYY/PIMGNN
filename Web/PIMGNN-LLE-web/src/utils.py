# -*- coding: utf-8 -*-
"""
utils.py (完整可覆盖版)

目标：
- 修复 ImportError: cannot import name 'renorm3' / 'set_seed' / 'batch_to_device'
- 提供工程所需的全部公共函数/类：
  set_seed, canonicalize_smiles, morgan_fp, renorm3, safe_group_apply_t, Scaler,
  atom_feature_dim, bond_feature_dim, global_feature_dim, mix_node_feature_dim, mix_edge_feature_dim,
  smiles_to_graph, batch_graphs, batch_to_device,
  build_mixture_graph, batch_mixture_graphs
- 兼容旧版 data.py 可能传入的 build_mixture_graph(dtype=...) 参数
"""

import random
import warnings
import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Crippen


# ============================================================
# Reproducibility / small helpers
# ============================================================

def set_seed(seed: int = 42) -> None:
    """全局设定随机种子，保证Torch/NumPy/随机库结果可重复。"""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonicalize_smiles(smi: str) -> str:
    """将SMILES标准化为RDKit规范形式，非法输入返回空字符串。"""
    if not isinstance(smi, str) or not smi.strip():
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """生成 Morgan 指纹（位向量），返回 float32 数组，非法 SMILES 返回零向量。"""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def renorm3(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """将三个组分重新归一化到和为1；总和过小则返回均分向量。"""
    x = np.asarray(a, dtype=np.float32).reshape(3,)
    x = np.clip(x, 0.0, None)
    s = float(np.sum(x))
    if s < eps:
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)
    return (x / s).astype(np.float32)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        v = float(v)
        if np.isnan(v) or np.isinf(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


# ============================================================
# t assignment for each (system_id, T) group (PCA ordering)
# ============================================================

def assign_t_by_pca(group: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 (system_id, T) 组的6维(E+R)做PCA，按第一主成分排序并分配t∈[0,1]。
    """
    cols = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
    X = group[cols].to_numpy(dtype=float)
    X = X - X.mean(axis=0, keepdims=True)

    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        pc1 = vt[0]
        proj = X @ pc1
    except Exception:
        proj = X[:, 0]

    order = np.argsort(proj)
    n = len(group)
    t = np.empty((n,), dtype=np.float32)
    if n == 1:
        t[:] = 0.5
    else:
        t[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)

    out = group.copy()
    out["t"] = t

    # 某些 pandas 版本 groupby.apply 可能丢列，这里兜底
    if "system_id" not in out.columns or "T" not in out.columns:
        try:
            sid, TT = group.name
            if "system_id" not in out.columns:
                out["system_id"] = sid
            if "T" not in out.columns:
                out["T"] = TT
        except Exception:
            pass
    return out


def safe_group_apply_t(df: pd.DataFrame) -> pd.DataFrame:
    """在(system_id, T)粒度调用assign_t_by_pca，兼容不同pandas版本的groupby行为。"""
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        out = df.groupby(["system_id", "T"], group_keys=False).apply(
            assign_t_by_pca, include_groups=False
        )
    except TypeError:
        out = df.groupby(["system_id", "T"], group_keys=False).apply(assign_t_by_pca)

    out = out.reset_index(drop=True)
    if "system_id" not in out.columns or "T" not in out.columns:
        raise ValueError("After groupby-apply, required columns system_id/T are missing.")
    return out


# ============================================================
# Scaler (train.py 里用 Scaler.fit(...)，这里要兼容)
# ============================================================

@dataclass
class Scaler:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (self.std + 1e-12)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x * (self.std + 1e-12) + self.mean

    @staticmethod
    def fit(x: np.ndarray) -> "Scaler":
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        return Scaler(mean=float(np.mean(x)), std=float(np.std(x) + 1e-8))


# ============================================================
# RDKit -> Graph featurization (molecule graph)
# ============================================================

ATOM_ELEMENTS = [
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I",
    "Na", "K", "Li", "Ca", "Mg", "Al", "Fe", "Cu", "Zn",
]

_PAULING_EN = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
    3: 0.98, 11: 0.93, 12: 1.31, 13: 1.61, 19: 0.82, 20: 1.00,
    26: 1.83, 29: 1.90, 30: 1.65,
}

def _one_hot(x: Any, xs: List[Any]) -> List[int]:
    return [int(x == a) for a in xs]

def _get_pauling_en(atomic_num: int) -> float:
    return float(_PAULING_EN.get(int(atomic_num), 0.0))

def atom_feature_dim() -> int:
    # 元素 one-hot(+other) + 3 + 6 + 3 + 4 + 8 （与本文件中的实现一致）
    elem = len(ATOM_ELEMENTS) + 1
    return elem + 3 + 6 + 3 + 4 + 8

def bond_feature_dim() -> int:
    # 键型4 + (conj,inring)2 + stereo6 + (order,en_diff,bond_len)3
    return 4 + 2 + 6 + 3

def global_feature_dim() -> int:
    # mol_global_features 输出长度
    return 15

def mix_node_feature_dim() -> int:
    # 改为返回全部全局特征维数（15）
    return 15

def mix_edge_feature_dim() -> int:
    return 16


def mol_global_features(mol: Chem.Mol) -> np.ndarray:
    """图级特征（Graph-level），单个分子一行。"""
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    mr = Crippen.MolMR(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ring = rdMolDescriptors.CalcNumRings(mol)
    arom_ring = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    formal_charge = sum(int(a.GetFormalCharge()) for a in mol.GetAtoms())
    charges = []
    for a in mol.GetAtoms():
        if a.HasProp("_GasteigerCharge"):
            charges.append(_safe_float(a.GetProp("_GasteigerCharge"), 0.0))
    if len(charges) == 0:
        q_mean = 0.0
        q_abs_mean = 0.0
        q_abs_max = 0.0
    else:
        q = np.asarray(charges, dtype=np.float32)
        q_mean = float(np.mean(q))
        q_abs_mean = float(np.mean(np.abs(q)))
        q_abs_max = float(np.max(np.abs(q)))

    feats = np.array([
        mw / 500.0,
        logp / 10.0,
        mr / 200.0,
        tpsa / 200.0,
        hbd / 10.0,
        hba / 20.0,
        rot / 20.0,
        ring / 20.0,
        arom_ring / 20.0,
        heavy / 100.0,
        frac_csp3,
        formal_charge / 5.0,
        q_mean,
        q_abs_mean,
        q_abs_max,
    ], dtype=np.float32)
    return feats


def _maybe_add_3d(mol: Chem.Mol, seed: int = 0) -> Tuple[Chem.Mol, Optional[Chem.Conformer]]:
    """Add a 3D conformer and optimize (fast-ish)."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    ok = AllChem.EmbedMolecule(mol, params)
    if ok != 0:
        params.randomSeed = int(seed) + 13
        ok = AllChem.EmbedMolecule(mol, params)

    conf = None
    if ok == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        try:
            conf = mol.GetConformer()
        except Exception:
            conf = None

    mol = Chem.RemoveHs(mol)
    return mol, conf


def smiles_to_graph(
    smiles: str,
    add_hs: bool = False,
    add_3d: bool = False,
    use_gasteiger: bool = True,
    max_atoms: int = 256,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """将SMILES转换为图字典，包含节点/边特征和全局特征；失败时返回占位空图。"""
    smi = canonicalize_smiles(smiles)
    if not smi:
        x = np.zeros((1, atom_feature_dim()), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, bond_feature_dim()), dtype=np.float32)
        g = np.zeros((global_feature_dim(),), dtype=np.float32)
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "g": g}

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        x = np.zeros((1, atom_feature_dim()), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, bond_feature_dim()), dtype=np.float32)
        g = np.zeros((global_feature_dim(),), dtype=np.float32)
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "g": g}

    if add_hs:
        mol = Chem.AddHs(mol)

    conf = None
    if add_3d:
        try:
            mol, conf = _maybe_add_3d(mol, seed=seed)
        except Exception:
            conf = None

    if use_gasteiger:
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            pass

    n_atoms = mol.GetNumAtoms()
    if n_atoms <= 0 or n_atoms > int(max_atoms):
        x = np.zeros((1, atom_feature_dim()), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, bond_feature_dim()), dtype=np.float32)
        g = np.zeros((global_feature_dim(),), dtype=np.float32)
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "g": g}

    pt = Chem.GetPeriodicTable()

    xs = []
    for i, atom in enumerate(mol.GetAtoms()):
        Z = atom.GetAtomicNum()
        elem_oh = _one_hot(atom.GetSymbol(), ATOM_ELEMENTS) + [int(atom.GetSymbol() not in ATOM_ELEMENTS)]
        deg = float(atom.GetDegree()) / 8.0
        formal = float(atom.GetFormalCharge()) / 5.0
        total_h = float(atom.GetTotalNumHs(includeNeighbors=True)) / 8.0

        hyb = atom.GetHybridization()
        hyb_list = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            "OTHER",
        ]
        hyb_val = hyb if hyb in hyb_list[:-1] else "OTHER"
        hyb_oh = _one_hot(hyb_val, hyb_list)

        arom = float(atom.GetIsAromatic())
        in_ring = float(atom.IsInRing())

        chir_list = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER,
        ]
        chir_oh = _one_hot(atom.GetChiralTag(), chir_list)

        mass = float(atom.GetMass()) / 200.0
        en = _get_pauling_en(Z) / 4.0
        rcov = float(pt.GetRcovalent(Z)) / 2.0
        rvdw = float(pt.GetRvdw(Z)) / 3.0

        q = 0.0
        if atom.HasProp("_GasteigerCharge"):
            q = _safe_float(atom.GetProp("_GasteigerCharge"), 0.0)
        q = float(np.clip(q, -1.0, 1.0))
        q_scaled = q

        # Crippen atom contributions
        try:
            contribs = Crippen._GetAtomContribs(mol)  # (logP, MR)
            logp_i, mr_i = contribs[i]
        except Exception:
            logp_i, mr_i = 0.0, 0.0
        logp_i = float(logp_i) / 5.0
        mr_i = float(mr_i) / 50.0

        # TPSA atom contribution (approx by fragment)
        tpsa_i = 0.0
        try:
            # 这里用分子级tpsa缩放作为弱替代（避免复杂fragment分解）
            tpsa_i = float(rdMolDescriptors.CalcTPSA(mol)) / max(1.0, float(n_atoms))
        except Exception:
            tpsa_i = 0.0
        tpsa_i = float(tpsa_i) / 50.0

        feat = (
            elem_oh
            + [float(Z) / 100.0, deg, formal]
            + hyb_oh
            + [arom, total_h, in_ring]
            + chir_oh
            + [mass, en, rcov, rvdw, q_scaled, logp_i, mr_i, tpsa_i]
        )
        xs.append(feat)

    x = np.asarray(xs, dtype=np.float32)

    # edges
    e_src, e_dst, e_attr = [], [], []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bt_oh = _one_hot(bt, [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ])
        conj = float(bond.GetIsConjugated())
        in_ring = float(bond.IsInRing())

        stereo = bond.GetStereo()
        stereo_oh = _one_hot(stereo, [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
        ])

        if bt == Chem.rdchem.BondType.SINGLE:
            order = 1.0
        elif bt == Chem.rdchem.BondType.DOUBLE:
            order = 2.0
        elif bt == Chem.rdchem.BondType.TRIPLE:
            order = 3.0
        elif bt == Chem.rdchem.BondType.AROMATIC:
            order = 1.5
        else:
            order = 0.0

        en_a = _get_pauling_en(mol.GetAtomWithIdx(a).GetAtomicNum())
        en_b = _get_pauling_en(mol.GetAtomWithIdx(b).GetAtomicNum())
        en_diff = abs(en_a - en_b)

        if conf is not None:
            pa = conf.GetAtomPosition(a)
            pb = conf.GetAtomPosition(b)
            bl = float(((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2) ** 0.5)
        else:
            bl = float(pt.GetRcovalent(mol.GetAtomWithIdx(a).GetAtomicNum()) +
                       pt.GetRcovalent(mol.GetAtomWithIdx(b).GetAtomicNum()))

        feat = bt_oh + [conj, in_ring] + stereo_oh + [order / 3.0, en_diff / 4.0, bl / 3.0]
        e_src += [a, b]
        e_dst += [b, a]
        e_attr += [feat, feat]

    edge_index = np.asarray([e_src, e_dst], dtype=np.int64)
    edge_attr = np.asarray(e_attr, dtype=np.float32) if len(e_attr) > 0 else np.zeros((0, bond_feature_dim()), dtype=np.float32)

    g = mol_global_features(mol)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "g": g}


# ============================================================
# Graph batching helpers (no PyG dependency)
# ============================================================

GraphDict = Dict[str, Any]

def batch_graphs(graphs: List[GraphDict]) -> GraphDict:
    """
    手写图批处理，不依赖PyG：
    - 拼接节点 x, 边 edge_index, edge_attr
    - 生成 batch (N,) 指示每个节点属于哪个图
    - 拼接图级特征 g -> (B, G)
    """
    import torch

    xs, edge_indices, edge_attrs, batches, gs = [], [], [], [], []
    node_offset = 0

    for i, g in enumerate(graphs):
        x = torch.as_tensor(g["x"], dtype=torch.float32)
        ei = torch.as_tensor(g["edge_index"], dtype=torch.long)
        ea = torch.as_tensor(g["edge_attr"], dtype=torch.float32)
        gg = torch.as_tensor(g.get("g", np.zeros((global_feature_dim(),), dtype=np.float32)), dtype=torch.float32).view(1, -1)

        n = x.shape[0]
        xs.append(x)
        gs.append(gg)
        batches.append(torch.full((n,), i, dtype=torch.long))

        if ei.numel() > 0:
            ei = ei + node_offset
            edge_indices.append(ei)
            edge_attrs.append(ea)

        node_offset += n

    X = torch.cat(xs, dim=0) if len(xs) else torch.zeros((0, atom_feature_dim()), dtype=torch.float32)
    batch = torch.cat(batches, dim=0) if len(batches) else torch.zeros((0,), dtype=torch.long)
    G = torch.cat(gs, dim=0) if len(gs) else torch.zeros((0, global_feature_dim()), dtype=torch.float32)

    if len(edge_indices) > 0:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, bond_feature_dim()), dtype=torch.float32)

    return {"x": X, "edge_index": edge_index, "edge_attr": edge_attr, "batch": batch, "g": G}


def batch_to_device(obj: Any, device: str) -> Any:
    """递归地将张量/容器移动到目标device，保持原始容器类型。"""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: batch_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        xs = [batch_to_device(v, device) for v in obj]
        return type(obj)(xs)
    return obj


# ============================================================
# Mixture graph (3 molecules) with geometry-driven interaction edges
# ============================================================

def _stable_int_seed(key: str, base_seed: int = 0) -> int:
    h = hashlib.md5((str(base_seed) + "|" + str(key)).encode("utf-8")).hexdigest()
    # 只取 31-bit，保证 Windows 下 RDKit 的 C long 不溢出
    return int(h[:8], 16) & 0x7FFFFFFF


def _random_rotation_matrix(rng: np.random.RandomState) -> np.ndarray:
    """Uniform random 3D rotation matrix (Shoemake)."""
    u1, u2, u3 = rng.rand(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    R = np.array([
        [1 - 2*(q3*q3 + q4*q4),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
        [    2*(q2*q3 + q1*q4), 1 - 2*(q2*q2 + q4*q4),     2*(q3*q4 - q1*q2)],
        [    2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2*q2 + q3*q3)],
    ], dtype=np.float32)
    return R

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def smiles_to_3d_package(smiles: str, seed: int = 0) -> Dict[str, Any]:
    """
    将单个分子转为一个用于几何交互计算的包：
    - coords: (N,3) 3D 坐标（尽力生成）
    - charges: (N,) Gasteiger 电荷（尽力计算）
    - center: (3,) 坐标中心
    - size: 近似分子“尺度”
    """
    smi = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return {"coords": np.zeros((1, 3), np.float32), "charges": np.zeros((1,), np.float32),
                "center": np.zeros((3,), np.float32), "size": 1.0}

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    mol3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    s = int(seed) & 0x7FFFFFFF
    params.randomSeed = s
    ok = AllChem.EmbedMolecule(mol, params)
    if ok != 0:
        params.randomSeed = (s + 13) & 0x7FFFFFFF
        ok = AllChem.EmbedMolecule(mol, params)


    if ok == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol3d, maxIters=200)
        except Exception:
            pass

    conf = None
    try:
        conf = mol3d.GetConformer()
    except Exception:
        conf = None

    n = mol3d.GetNumAtoms()
    coords = np.zeros((n, 3), dtype=np.float32)
    if conf is not None:
        for i in range(n):
            p = conf.GetAtomPosition(i)
            coords[i] = np.array([p.x, p.y, p.z], dtype=np.float32)

    charges = np.zeros((n,), dtype=np.float32)
    for i, a in enumerate(mol3d.GetAtoms()):
        if a.HasProp("_GasteigerCharge"):
            charges[i] = float(np.clip(_safe_float(a.GetProp("_GasteigerCharge"), 0.0), -1.0, 1.0))

    center = coords.mean(axis=0, keepdims=False)
    coords0 = coords - center[None, :]
    size = float(np.sqrt(np.mean(np.sum(coords0 * coords0, axis=1))) + 1e-6)

    return {"coords": coords, "charges": charges, "center": center.astype(np.float32), "size": size}


def pair_interaction_features_3d(pkg_a: Dict[str, Any], pkg_b: Dict[str, Any], T: float,
                                 seed: int = 0) -> np.ndarray:
    """
    计算两个分子间的“交互边特征” (mix_edge_feature_dim=16)
    纯启发式：距离/方向/电荷统计/温度缩放等
    """
    rng = np.random.RandomState(int(seed))
    Ra = _random_rotation_matrix(rng)
    Rb = _random_rotation_matrix(rng)

    ca = (pkg_a["coords"] - pkg_a["center"][None, :]) @ Ra.T
    cb = (pkg_b["coords"] - pkg_b["center"][None, :]) @ Rb.T
    qa = pkg_a["charges"]
    qb = pkg_b["charges"]

    # center-center vector
    v = (pkg_b["center"] - pkg_a["center"]).astype(np.float32)
    dist_cent = float(np.linalg.norm(v) + 1e-6)
    dir_v = _normalize(v)

    # min atom-atom distance (approx)
    # 为了别太慢，只采样少量点
    idx_a = np.linspace(0, ca.shape[0]-1, num=min(16, ca.shape[0]), dtype=int)
    idx_b = np.linspace(0, cb.shape[0]-1, num=min(16, cb.shape[0]), dtype=int)
    da = ca[idx_a]
    db = cb[idx_b]
    # (na, nb, 3)
    diff = da[:, None, :] - db[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    min_d = float(np.sqrt(np.min(d2) + 1e-6))
    mean_d = float(np.sqrt(np.mean(d2) + 1e-6))

    # charge interaction stats (coarse)
    qa_s = qa[idx_a] if qa.shape[0] > 0 else np.zeros((len(idx_a),), np.float32)
    qb_s = qb[idx_b] if qb.shape[0] > 0 else np.zeros((len(idx_b),), np.float32)
    qprod = float(np.mean(qa_s[:, None] * qb_s[None, :]))
    qabs = float(np.mean(np.abs(qa_s[:, None] * qb_s[None, :])))

    # size stats
    sa = float(pkg_a.get("size", 1.0))
    sb = float(pkg_b.get("size", 1.0))

    # temperature features
    T = float(T)
    T1 = T / 500.0
    T2 = (T * T) / (500.0 * 500.0)

    feats = np.array([
        dist_cent / 10.0,      # 0
        min_d / 10.0,          # 1
        mean_d / 10.0,         # 2
        dir_v[0],              # 3
        dir_v[1],              # 4
        dir_v[2],              # 5
        sa / 10.0,             # 6
        sb / 10.0,             # 7
        (sa + sb) / 20.0,      # 8
        abs(sa - sb) / 10.0,   # 9
        qprod,                 # 10
        qabs,                  # 11
        float(np.mean(qa_s)),  # 12
        float(np.mean(qb_s)),  # 13
        T1,                    # 14
        T2,                    # 15
    ], dtype=np.float32)

    return feats


def build_mixture_graph(
    smiles1: str,
    smiles2: str,
    smiles3: str,
    T: float = None,
    seed: int = 0,
    dtype: Any = None,  # 兼容旧版本传 dtype
    # ---- 兼容 data.py 传入的关键字参数 ----
    T_norm: float = None,
    T_raw: float = None,
    cfg: Any = None,
    mol_cache: Optional[Dict[str, Any]] = None,
    pair_cache: Optional[Dict[str, Any]] = None,
    **kwargs,  # 吞掉未来可能加的参数，避免再炸
) -> Dict[str, np.ndarray]:
    """
    兼容 data.py 的调用方式：
      build_mixture_graph(..., T_norm=..., T_raw=..., cfg=..., mol_cache=..., pair_cache=...)

    返回:
      {'x': (3,F), 'edge_index': (2,6), 'edge_attr': (6,De)}
    """

    # ---- 1) 温度选择：优先用 T_raw（data.py 一定会传）----
    if T_raw is not None:
        T_use = float(T_raw)
    elif T is not None:
        T_use = float(T)
    elif T_norm is not None and cfg is not None and hasattr(cfg, "T_MEAN") and hasattr(cfg, "T_STD"):
        # 兜底：如果只传了 T_norm，且 cfg 有均值方差，尝试反归一化
        T_use = float(T_norm) * float(getattr(cfg, "T_STD")) + float(getattr(cfg, "T_MEAN"))
    else:
        T_use = 298.15  # 兜底常温

    # ---- 2) canonical smiles 用于 cache key（避免同一分子重复算）----
    s1c = canonicalize_smiles(smiles1)
    s2c = canonicalize_smiles(smiles2)
    s3c = canonicalize_smiles(smiles3)

    # ---- 3) node features：沿用你原先的 10 维（分子全局特征前10维）----
    def _node_feat(smi: str) -> np.ndarray:
        smi_c = canonicalize_smiles(smi)
        mol = Chem.MolFromSmiles(smi_c) if smi_c else None
        if mol is None:
            return np.zeros((mix_node_feature_dim(),), dtype=np.float32)
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            pass
        g = mol_global_features(mol)  # 15
        f = g.astype(np.float32, copy=False)  # 直接用全部全局特征
        if f.shape[0] < mix_node_feature_dim():
            pad = np.zeros((mix_node_feature_dim() - f.shape[0],), dtype=np.float32)
            f = np.concatenate([f, pad], axis=0)
        return f

    x = np.stack([_node_feat(smiles1), _node_feat(smiles2), _node_feat(smiles3)], axis=0).astype(np.float32)

    # ---- 4) mol_cache：缓存 3D package（最省时间的地方）----
    if mol_cache is None:
        mol_cache = {}

    def _get_pkg(smi: str, smi_c: str, tag: str) -> Dict[str, Any]:
        if smi_c and smi_c in mol_cache:
            return mol_cache[smi_c]
        pkg = smiles_to_3d_package(smi, seed=_stable_int_seed(f"{seed}|pkg|{tag}", seed))
        if smi_c:
            mol_cache[smi_c] = pkg
        return pkg

    pkg1 = _get_pkg(smiles1, s1c, "a")
    pkg2 = _get_pkg(smiles2, s2c, "b")
    pkg3 = _get_pkg(smiles3, s3c, "c")

    # ---- 5) pair_cache：缓存有向 pair 交互特征（同一 pair + 温度重复出现很多）----
    if pair_cache is None:
        pair_cache = {}

    def _pair_key(sa: str, sb: str, Tval: float) -> str:
        # ordered pair key（有向）
        return f"{sa}||{sb}||{float(Tval):.6f}"

    def _get_pair_feat(pkg_a: Dict[str, Any], pkg_b: Dict[str, Any],
                       sa: str, sb: str, tag: str) -> np.ndarray:
        k = _pair_key(sa, sb, T_use)
        hit = pair_cache.get(k, None)
        if hit is not None:
            return hit
        feat = pair_interaction_features_3d(pkg_a, pkg_b, T=T_use, seed=_stable_int_seed(f"{seed}|pair|{tag}", seed))
        feat = np.asarray(feat, dtype=np.float32)
        pair_cache[k] = feat
        return feat

    # ---- 6) build fully-connected directed edges among 3 nodes ----
    pairs = [
        (0, 1, pkg1, pkg2, s1c, s2c, "01"),
        (0, 2, pkg1, pkg3, s1c, s3c, "02"),
        (1, 2, pkg2, pkg3, s2c, s3c, "12"),
    ]

    e_src, e_dst, e_attr = [], [], []
    for u, v, pu, pv, su, sv, tag in pairs:
        feat_uv = _get_pair_feat(pu, pv, su, sv, tag)
        feat_vu = _get_pair_feat(pv, pu, sv, su, tag + "r")
        e_src += [u, v]
        e_dst += [v, u]
        e_attr += [feat_uv.tolist(), feat_vu.tolist()]

    edge_index = np.asarray([e_src, e_dst], dtype=np.int64)
    edge_attr = np.asarray(e_attr, dtype=np.float32) if len(e_attr) else np.zeros((0, mix_edge_feature_dim()), dtype=np.float32)

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}



def batch_mixture_graphs(graphs: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """batch mixture graphs（每图3节点）"""
    import torch

    xs, edge_indices, edge_attrs, batches = [], [], [], []
    node_offset = 0

    for i, g in enumerate(graphs):
        x = torch.as_tensor(g["x"], dtype=torch.float32)
        ei = torch.as_tensor(g["edge_index"], dtype=torch.long)
        ea = torch.as_tensor(g["edge_attr"], dtype=torch.float32)

        n = x.shape[0]
        xs.append(x)
        batches.append(torch.full((n,), i, dtype=torch.long))

        if ei.numel() > 0:
            edge_indices.append(ei + node_offset)
            edge_attrs.append(ea)

        node_offset += n

    X = torch.cat(xs, dim=0) if xs else torch.zeros((0, mix_node_feature_dim()), dtype=torch.float32)
    batch = torch.cat(batches, dim=0) if batches else torch.zeros((0,), dtype=torch.long)

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, mix_edge_feature_dim()), dtype=torch.float32)

    return {"x": X, "edge_index": edge_index, "edge_attr": edge_attr, "batch": batch}


# ============================================================
# Functional-group (FG) extraction (RDKit SMARTS)
# ============================================================
# NOTE:
# - This is intentionally lightweight (no 3D needed).
# - Output is a set of fragment SMILES strings (canonicalized by RDKit).
# - Used to build a fixed-length multi-hot vector (FG_TOPK).
from rdkit import Chem as _Chem

_FG_PATT = {
    "HETEROATOM": "[!#6]",
    "DOUBLE_TRIPLE_BOND": "*=,#*",
    "ACETAL": "[CX4]([O,N,S])[O,N,S]",
}
_FG_PATT = {k: _Chem.MolFromSmarts(v) for k, v in _FG_PATT.items()}


def get_fg_set(mol: "_Chem.Mol") -> set:
    """Identify functional groups and rings and return as a set of fragment SMILES.

    This implementation is adapted from common FG-mining pipelines:
    - mark functional atoms by SMARTS
    - merge connected marked atoms
    - include their 1-hop environments
    - include non-ring single bonds as trivial fragments
    - merge fused rings and include rings as fragments

    Args:
        mol: RDKit Mol
    Returns:
        set of fragment SMILES strings
    """
    fgs = []

    # ---- identify and merge rings (fused rings) ----
    rings = [set(x) for x in _Chem.GetSymmSSSR(mol)]
    flag = True
    while flag:
        flag = False
        for i in range(len(rings)):
            if len(rings[i]) == 0:
                continue
            for j in range(i + 1, len(rings)):
                shared_atoms = rings[i] & rings[j]
                if len(shared_atoms) > 2:
                    rings[i].update(rings[j])
                    rings[j] = set()
                    flag = True
    rings = [r for r in rings if len(r) > 0]

    # ---- identify functional atoms and merge connected ones ----
    marks = set()
    for patt in _FG_PATT.values():
        if patt is None:
            continue
        for sub in mol.GetSubstructMatches(patt):
            marks.update(sub)

    atom2fg = [[] for _ in range(mol.GetNumAtoms())]
    for atom in marks:
        fgs.append({atom})
        atom2fg[atom] = [len(fgs) - 1]

    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in marks and a2 in marks:
            # merge their FGs
            if len(atom2fg[a1]) != 1 or len(atom2fg[a2]) != 1:
                # extremely rare, but keep robust
                continue
            fgs[atom2fg[a1][0]].update(fgs[atom2fg[a2][0]])
            fgs[atom2fg[a2][0]] = set()
            atom2fg[a2] = atom2fg[a1]
        elif a1 in marks:
            if len(atom2fg[a1]) == 1:
                fgs[atom2fg[a1][0]].add(a2)
                atom2fg[a2].extend(atom2fg[a1])
        elif a2 in marks:
            if len(atom2fg[a2]) == 1:
                fgs[atom2fg[a2][0]].add(a1)
                atom2fg[a1].extend(atom2fg[a2])
        else:
            # trivial single bond fragment
            fgs.append({a1, a2})
            atom2fg[a1].append(len(fgs) - 1)
            atom2fg[a2].append(len(fgs) - 1)

    # filter
    tmp = []
    for fg in fgs:
        if len(fg) == 0:
            continue
        if len(fg) == 1 and mol.GetAtomWithIdx(list(fg)[0]).IsInRing():
            continue
        tmp.append(fg)
    fgs = tmp

    # final FGs: rings + non-ring FGs
    fgs.extend(rings)

    fg_smiles = set()
    for fg in fgs:
        try:
            fg_smiles.add(_Chem.MolFragmentToSmiles(mol, list(fg), canonical=True))
        except Exception:
            # be robust: skip problematic fragments
            continue
    return fg_smiles


def fg_smiles_from_smiles(smiles: str) -> set:
    """Return a set of FG fragment SMILES extracted from a SMILES string."""
    if smiles is None:
        return set()
    smiles = str(smiles).strip()
    if smiles == "" or smiles.lower() == "nan":
        return set()
    try:
        mol = _Chem.MolFromSmiles(smiles)
        if mol is None:
            return set()
        return get_fg_set(mol)
    except Exception:
        return set()
