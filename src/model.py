# -*- coding: utf-8 -*-
"""
D:\GGNN\YXFL\src\model.py  (完整可覆盖)

兼容三种模式：
1) 指纹（FP）: x 是 Tensor (B, 3*FP_BITS+2)
2) 图拼接（Graph-concat）: x 是 dict {'g1','g2','g3','scalars'}
3) 混合物图（Mixture-graph）: 在图拼接基础上，x 可额外包含 {'mix'}，
   用 mix 的 edge_attr/edge_index 做分子级 message passing，并输出 mix_emb。

本工程的“图”是自定义 GraphDict（不依赖 PyG），由 utils.batch_graphs 生成：
  g = {'x': (N,F), 'edge_index': (2,E), 'edge_attr': (E,De), 'batch': (N,), 'g': (B,G)}
"""
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from utils import atom_feature_dim, bond_feature_dim, global_feature_dim


# ----------------------------
# pooling helpers (no torch_scatter required)
# ----------------------------
def global_pool_mean(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """
    安全的全局平均池化：将节点特征聚合为图特征
    
    Args:
        x: 节点特征张量，形状 (N, F)，N 为节点总数
        batch: 节点到图的映射，形状 (N,)，值域 [0, num_graphs)
        num_graphs: 图的总数 B
    
    Returns:
        池化后的图特征，形状 (B, F)
    """
    if x.numel() == 0:
        return torch.zeros((num_graphs, x.size(-1)), device=x.device, dtype=x.dtype)

    out = torch.zeros((num_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt = torch.zeros((num_graphs,), device=x.device, dtype=x.dtype)
    one = torch.ones((batch.size(0),), device=x.device, dtype=x.dtype)
    cnt.index_add_(0, batch, one)
    cnt = cnt.clamp_min(1.0).unsqueeze(-1)
    return out / cnt


def _num_graphs_from_batch(batch: torch.Tensor, fallback: int) -> int:
    """
    从 batch 张量推断图的总数
    
    Args:
        batch: 节点到图的映射张量
        fallback: 当 batch 为空时的默认值
    
    Returns:
        图的总数
    """
    if batch.numel() == 0:
        return int(fallback)
    return int(batch.max().item()) + 1


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    计算掩码平均值（用于处理变长序列）
    
    Args:
        x: 输入张量，形状通常为 (B, L, D)（批、长度、维度）
        mask: 掩码张量，1 表示有效，0 表示填充，形状 (B, L) 或 (B, L, 1)
    
    Returns:
        平均特征，形状 (B, D)
    """
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(dtype=x.dtype)
    if m.dim() == 2:
        m = m.unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom


def s3_equivariant_embedding(mol_embedding: torch.Tensor) -> torch.Tensor:
    """
    S3 置换等变处理：为三个分子的嵌入添加对称背景
    
    用于对称分子系统（三元混合物）的等变性处理。计算三个分子的平均嵌入作为对称背景，
    并将其加回到每个分子的嵌入中，保证在分子排列变换下的等变性。
    
    Args:
        mol_embedding: 分子嵌入，可以是：
            - (B, 3, H)：三维张量，B 为批大小，3 为分子数，H 为隐藏维度
            - 3 个 (B, H) 张量的列表或元组
    
    Returns:
        处理后的嵌入，形状 (B, 3, H)
        每个分子的嵌入 = 原嵌入 + 三个分子嵌入的平均值
    """
    if isinstance(mol_embedding, (list, tuple)):
        mol_embedding = torch.stack(mol_embedding, dim=1)
    if mol_embedding.ndim != 3 or mol_embedding.shape[1] != 3:
        raise ValueError(f"mol_embedding must be (B,3,H), got {tuple(mol_embedding.shape)}")
    # 计算三个分子的平均嵌入作为对称背景
    mean = mol_embedding.mean(dim=1, keepdim=True)
    # 将对称背景加回到每个分子的嵌入中
    return mol_embedding + mean


def cross_molecular_fg_attention(
    fg1: torch.Tensor,
    fg2: torch.Tensor,
    fg3: torch.Tensor,
    mask1: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
    mask3: Optional[torch.Tensor] = None,
    attn: Optional[nn.MultiheadAttention] = None,
    norm: Optional[nn.LayerNorm] = None,
    drop: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    跨分子功能基团（FG）注意力机制
    
    对三个分子的 FG token 序列进行交互注意力计算。每个分子的 FG 序列与其他两个分子的
    FG 序列进行交互，通过多头注意力机制捕捉分子间的功能基团相互作用。
    
    Args:
        fg1, fg2, fg3: FG token 嵌入，形状 (B, L, D)，B 为批大小，L 为序列长度，D 为嵌入维度
        mask1, mask2, mask3: FG token 掩码，形状 (B, L)，1 表示有效 token，0 表示填充
        attn: 多头注意力层（nn.MultiheadAttention），若为 None 则退化为简单平均
        norm: 层标准化层（nn.LayerNorm）
        drop: 随机丢弃层（nn.Dropout）
    
    Returns:
        三元组 (p1, p2, p3)，每个都是池化后的 FG 表示，形状 (B, D)
    """
    # 若没有注意力层，则直接计算掩码平均
    if attn is None:
        return _masked_mean(fg1, mask1), _masked_mean(fg2, mask2), _masked_mean(fg3, mask3)

    # 准备键值对：将两个 FG 序列拼接，并处理掩码
    def _prep_kv(a: torch.Tensor, b: torch.Tensor, ma: Optional[torch.Tensor], mb: Optional[torch.Tensor]):
        kv = torch.cat([a, b], dim=1)
        if ma is None or mb is None:
            return kv, None
        m = torch.cat([ma, mb], dim=1).to(dtype=torch.bool)
        key_padding_mask = ~m
        # 如果某个样本的所有键都是填充，将第一个键设为有效（避免注意力全为 0）
        if key_padding_mask.all(dim=1).any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[key_padding_mask.all(dim=1), 0] = False
        return kv, key_padding_mask

    # 注意力前向传递
    def _attend(q: torch.Tensor, kv: torch.Tensor, q_mask: Optional[torch.Tensor], kv_mask: Optional[torch.Tensor]):
        # 多头注意力：查询与键值对交互
        out, _ = attn(q, kv, kv, key_padding_mask=kv_mask, need_weights=False)
        if drop is not None:
            out = drop(out)
        # 残差连接
        out = q + out
        if norm is not None:
            out = norm(out)
        # 应用查询掩码
        if q_mask is not None:
            out = out * q_mask.to(dtype=out.dtype).unsqueeze(-1)
        return out

    # 为三个分子构建交互对
    kv23, m23 = _prep_kv(fg2, fg3, mask2, mask3)  # fg1 与 fg2、fg3 交互
    kv13, m13 = _prep_kv(fg1, fg3, mask1, mask3)  # fg2 与 fg1、fg3 交互
    kv12, m12 = _prep_kv(fg1, fg2, mask1, mask2)  # fg3 与 fg1、fg2 交互

    # 执行交叉注意力
    a1 = _attend(fg1, kv23, mask1, m23)
    a2 = _attend(fg2, kv13, mask2, m13)
    a3 = _attend(fg3, kv12, mask3, m12)

    # 池化并返回
    return _masked_mean(a1, mask1), _masked_mean(a2, mask2), _masked_mean(a3, mask3)


# ----------------------------
# Fingerprint baseline (kept)
# ----------------------------
class LLECurveNet(nn.Module):
    """
    指纹基线模型 - LLE 曲线预测
    
    将三个分子的摩根指纹、温度和组成输入，预测液液平衡性质（富集因子和相分离线）。
    
    模型架构：
        - backbone：两层全连接网络 (in_dim → hidden → hidden)，使用 GELU 激活
        - head_E：线性层预测三个分子的富集因子 E1, E2, E3（输出维度 3）
        - head_R：线性层预测三个分子的分配比 R1, R2, R3（输出维度 3）
    
    两个输出头都应用 softmax 约束以强制分布性质 (E1+E2+E3=1, R1+R2+R3=1)。
    
    Args:
        in_dim: 输入特征维度（Morgan FP × 3 + T_norm + t）
        hidden: 隐层维度，默认 1024
        dropout: 丢弃率，默认 0.15
    
    输入形状: (B, in_dim)，B 为批大小
    输出形状: (B, 6)，排列为 [E1, E2, E3, R1, R2, R3]，每个都在 (0,1) 范围内且行和为 1
    """
    def __init__(self, in_dim: int, hidden: int = 1024, dropout: float = 0.15):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_E = nn.Linear(hidden, 3)
        self.head_R = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：特征 → 隐层 → [富集因子, 分配比]"""
        h = self.backbone(x)
        # 软最大值约束确保每个输出向量求和为 1
        E = torch.softmax(self.head_E(h), dim=-1)
        R = torch.softmax(self.head_R(h), dim=-1)
        return torch.cat([E, R], dim=-1)


# ----------------------------
# Molecule graph encoder (custom graph dict)
# ----------------------------
class MPNNLayer(nn.Module):
    """
    消息传递神经网络层 - 图节点和边迭代更新
    
    该层通过消息传递机制实现原子节点的状态更新，并支持边属性的迭代细化。核心特性：
    
    1. **节点更新**：源节点 h[src] 与边属性进行消息计算，聚合到目标节点 h[dst]
    2. **边更新**（可选）：源+目标节点信息与边属性进行融合更新，使用门控机制和残差缩放
    3. **边丢弃**：训练中应用边级随机丢弃，同时影响消息传递和边更新
    4. **残差连接**：节点和边都使用残差缩放以稳定深层网络训练
    
    Args:
        hidden: 节点隐层维度（通常 256/512）
        edge_dim: 边属性维度（通常 10-20）
        dropout: MLP 丢弃率，默认 0.15
        update_edges: 是否进行边属性迭代更新，默认 True
        edge_dropout: 边级丢弃率，默认 0.1（防止共适应）
        edge_scale: 边更新残差缩放系数，默认 0.1（控制更新幅度）
    
    前向传播流程：
        1. 提取源和目标节点的隐状态
        2. 生成边丢弃掩码（训练时）
        3. 若启用边更新：计算 [h_src, h_dst, e] 的融合，应用门控和残差
        4. 消息计算：[h_src, e] → 消息，应用丢弃掩码
        5. 消息聚合：按目标节点索引求和
        6. 节点更新：[h_old, agg_msg] → h_new，应用残差和层标准化
        7. 返回更新后的节点和边
    """
    def __init__(
        self,
        hidden: int,
        edge_dim: int,
        dropout: float,
        update_edges: bool = True,
        edge_dropout: float = 0.1,
        edge_scale: float = 0.1,
    ):
        super().__init__()
        self.update_edges = bool(update_edges)
        self.edge_dropout = float(edge_dropout)
        self.edge_scale = float(edge_scale)

        # 消息网络：从源节点和边属性计算消息
        self.msg = nn.Sequential(
            nn.Linear(hidden + edge_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        # 节点更新网络：从原节点和聚合消息计算新状态
        self.upd = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

        if self.update_edges:
            # 边MLP：融合源+目标节点与边属性，输出边的增量更新
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden * 2 + edge_dim, edge_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(edge_dim, edge_dim),
            )
            # 边门控：决定边更新的强度
            self.edge_gate = nn.Sequential(
                nn.Linear(hidden * 2 + edge_dim, 1),
                nn.Sigmoid()
            )
            self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """
        前向传播
        
        Args:
            h: 节点隐状态，形状 (N, hidden)，N 为节点数
            edge_index: 边索引，形状 (2, E)，第一行为源节点，第二行为目标节点
            edge_attr: 边属性，形状 (E, edge_dim)，E 为边数
        
        Returns:
            更新后的节点隐状态 (N, hidden) 和边属性 (E, edge_dim)
        """
        # 处理空图的情况
        if edge_index.numel() == 0:
            return self.norm(h), edge_attr

        # AMP 精度对齐：确保边属性与节点隐状态数据类型一致
        if edge_attr.dtype != h.dtype:
            edge_attr = edge_attr.to(dtype=h.dtype)

        # 解包边索引：将边端点映射到对应节点
        src, dst = edge_index[0], edge_index[1]
        hs = h[src]  # (E, hidden) - 源节点隐状态
        hd = h[dst]  # (E, hidden) - 目标节点隐状态

        E = edge_attr.size(0)
        # 边丢弃掩码：训练时随机丢弃一些边，影响消息和边更新
        if self.training and self.edge_dropout > 0:
            keep = (torch.rand(E, device=edge_attr.device) > self.edge_dropout).to(edge_attr.dtype)
        else:
            keep = torch.ones(E, device=edge_attr.device, dtype=edge_attr.dtype)

        # ===== 边属性更新（如果启用）=====
        if self.update_edges:
            # 联合表示：源和目标节点信息与边属性
            z = torch.cat([hs, hd, edge_attr], dim=-1)  # (E, 2H+De)
            # 计算门控权重和增量更新
            gate = self.edge_gate(z)                    # (E, 1) - 范围 [0, 1]
            delta = self.edge_mlp(z)                    # (E, De) - 增量
            # 应用门控和边丢弃掩码，通过残差缩放添加到原边属性
            delta = delta * gate                        # 门控约束
            edge_attr = self.edge_norm(edge_attr + (self.edge_scale * keep.unsqueeze(-1) * delta))

        # ===== 消息传递 =====
        # 从源节点和边属性计算消息
        m = self.msg(torch.cat([hs, edge_attr], dim=-1))  # (E, hidden)
        # 应用边丢弃掩码：丢弃的边其消息置零
        m = m * keep.unsqueeze(-1)

        # ===== 消息聚合 =====
        # 按目标节点索引累加消息（图求和操作）
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)

        # ===== 节点更新 =====
        # 新状态 = 更新网络([旧状态, 聚合消息])
        h2 = self.upd(torch.cat([h, agg], dim=-1))
        # 残差连接 + 层标准化以稳定训练
        h = self.norm(h + h2)
        return h, edge_attr


class MPNNEncoder(nn.Module):
    """
    分子图编码器 - 将分子图转换为图级表示
    
    该编码器处理批量分子图，通过多层消息传递神经网络进行节点信息聚集，
    结合分子全局特征，最终输出每个分子的统一表示。
    
    输入图字典必须包含以下键：
        - x: 节点特征，形状 (N, node_dim)，N 为总节点数
        - edge_index: 边索引，形状 (2, E)，E 为总边数
        - edge_attr: 边属性，形状 (E, edge_dim)
        - batch: 节点所属图索引，形状 (N,)，范围 [0, B)，B 为批大小
        - g: 分子全局特征，形状 (B, global_dim)
    
    Args:
        node_dim: 输入节点特征维度（原子特征数）
        edge_dim: 输入边属性维度（键合类型等特征数）
        global_dim: 分子全局特征维度（分子描述符数）
        hidden: 隐层维度，默认 256
        layers: MPNN 层数，默认 4
        dropout: MLP 丢弃率，默认 0.15
        pool: 池化策略，目前仅支持 "mean"
        update_edges: 是否在 MPNNLayer 中更新边属性，默认 True
        edge_dropout: 边级丢弃率，默认 0.1
        edge_scale: 边更新残差缩放系数，默认 0.1
    
    前向传播流程：
        1. 节点特征投影：(node_dim → hidden)
        2. 全局特征投影：(global_dim → hidden)
        3. 多层 MPNN：迭代更新节点表示
        4. 节点池化：按 batch 索引计算每个图的平均节点表示
        5. 融合：拼接池化节点和投影全局特征
        6. 输出层：融合特征 → 最终图表示 (hidden)
    
    输入：图字典 Dict[str, Tensor]
    输出：每个分子的表示 (B, hidden)
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden: int = 256,
        layers: int = 4,
        dropout: float = 0.15,
        pool: str = "mean",
        update_edges: bool = True,
        edge_dropout: float = 0.1,
        edge_scale: float = 0.1,
    ):
        super().__init__()
        self.pool = pool

        # 节点特征投影层：原始特征 → 隐层维度
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 全局特征投影层：分子描述符 → 隐层维度
        self.glob_proj = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 堆叠多个 MPNN 层进行迭代消息传递
        self.layers = nn.ModuleList([
            MPNNLayer(
                hidden=hidden,
                edge_dim=edge_dim,
                dropout=dropout,
                update_edges=update_edges,
                edge_dropout=edge_dropout,
                edge_scale=edge_scale,
            )
            for _ in range(int(layers))
        ])

        # 输出层：融合节点和全局特征
        self.out = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, g: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            g: 图字典，包含 x, edge_index, edge_attr, batch, g
        
        Returns:
            每个分子的编码表示，形状 (B, hidden)
        """
        x = g["x"]
        edge_index = g["edge_index"]
        edge_attr = g["edge_attr"]
        batch = g["batch"]
        glob = g["g"]

        # 投影节点特征
        h = self.node_proj(x)

        # AMP 精度对齐：确保数据类型一致
        if edge_attr.dtype != h.dtype:
            edge_attr = edge_attr.to(dtype=h.dtype)
        if glob.dtype != h.dtype:
            glob = glob.to(dtype=h.dtype)

        # 多层 MPNN 迭代更新节点表示
        for layer in self.layers:
            h, edge_attr = layer(h, edge_index, edge_attr)

        # 获取批大小和图数量
        num_graphs = int(glob.shape[0])
        if num_graphs == 0:
            return torch.zeros((0, h.size(-1)), device=h.device, dtype=h.dtype)

        # 节点池化：计算每个图的平均节点表示
        pooled = global_pool_mean(h, batch, num_graphs)
        # 全局特征投影
        g_emb = self.glob_proj(glob)
        # 融合和输出
        z = self.out(torch.cat([pooled, g_emb], dim=-1))
        return z


# ----------------------------
# Mixture graph encoder (3 nodes per sample)
# ----------------------------
class MixGraphEncoder(nn.Module):
    """
    混合物图编码器 - 处理三元混合物的分子间相互作用
    
    该编码器在混合物图上执行消息传递，将分子嵌入作为节点特征。对于三元混合物，
    每个样本由 3 个节点（分别代表 3 个分子）和 3 条边（代表分子对间的相互作用）构成。
    
    核心功能：
    1. **边属性投影**：将原始混合物图边属性投影到隐层维度
    2. **分子间 MPNN**：多层消息传递更新分子表示，捕捉混合物特定的相互作用
    3. **图级池化**：计算每个混合物的聚合表示（通常 3 个分子特征的平均）
    
    输入混合物图字典：
        - edge_index: 边索引，形状 (2, 3)，通常为 [(0,1),(1,2),(0,2)] 三角形
        - edge_attr: 边属性，形状 (3, edge_attr_dim)，通常是混合物交互特征
        - batch: 节点所属图索引，形状 (3*B,)，B 为批大小
    
    Args:
        hidden: 节点隐层维度（通常与分子编码器相同）
        layers: 混合物图 MPNN 层数，默认 2
        dropout: MLP 丢弃率，默认 0.15
        edge_hidden: 混合物边属性投影维度，默认 64
        update_edges: 是否迭代更新边属性，默认 True
        edge_dropout: 边级丢弃率，默认 0.1
        edge_scale: 边更新残差缩放系数，默认 0.1
    
    前向传播流程：
        1. 检查边的有效性；若空则直接池化并返回
        2. 边属性投影：原始边维度 → edge_hidden，通过激活和标准化
        3. 多层 MPNN：在 3-节点图上传递消息，更新分子表示
        4. 输出标准化：应用层标准化
        5. 图级池化：计算每个混合物的分子特征平均
    
    Args（前向）:
        node_h: 分子嵌入，形状 (3*B, hidden)，通常是堆叠的 [e1, e2, e3]
        mix_g: 混合物图字典
        fallback_num_graphs: 若无法从 batch 推断图数，使用此备用值
    
    Returns:
        node_out: 更新后的分子表示，形状 (3*B, hidden)
        mix_emb: 每个混合物的聚合表示，形状 (B, hidden)
    """
    def __init__(
        self,
        hidden: int,
        layers: int = 2,
        dropout: float = 0.15,
        edge_hidden: int = 64,
        update_edges: bool = True,
        edge_dropout: float = 0.1,
        edge_scale: float = 0.1,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.edge_hidden = int(edge_hidden)

        # 懒加载线性层：自动适应输入边属性的任意维度
        self.edge_in = nn.LazyLinear(self.edge_hidden)
        self.edge_act = nn.GELU()
        self.edge_drop = nn.Dropout(dropout)
        self.edge_norm = nn.LayerNorm(self.edge_hidden)

        # 混合物图消息传递层
        self.layers = nn.ModuleList([
            MPNNLayer(
                hidden=self.hidden,
                edge_dim=self.edge_hidden,
                dropout=dropout,
                update_edges=update_edges,
                edge_dropout=edge_dropout,
                edge_scale=edge_scale,
            )
            for _ in range(int(layers))
        ])
        self.out_norm = nn.LayerNorm(self.hidden)

    def forward(
        self,
        node_h: torch.Tensor,
        mix_g: Dict[str, torch.Tensor],
        fallback_num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_h: 分子嵌入，形状 (3*B, hidden)
            mix_g: 混合物图字典，包含 edge_index, edge_attr, batch
            fallback_num_graphs: 备用图数量
        
        Returns:
            (node_out, mix_emb)
                - node_out: 更新后的分子表示 (3*B, hidden)
                - mix_emb: 混合物级聚合表示 (B, hidden)
        """
        edge_index = mix_g["edge_index"]
        edge_attr = mix_g["edge_attr"]
        batch = mix_g["batch"]

        # 处理空边的情况（某些混合物可能没有预定义的相互作用边）
        if edge_attr.numel() == 0 or edge_index.numel() == 0:
            num_graphs = _num_graphs_from_batch(batch, fallback=fallback_num_graphs)
            # 直接标准化和池化，无需消息传递
            node_out = self.out_norm(node_h)
            mix_emb = global_pool_mean(node_out, batch, num_graphs)
            return node_out, mix_emb

        # AMP 精度对齐
        if edge_attr.dtype != node_h.dtype:
            edge_attr = edge_attr.to(dtype=node_h.dtype)

        # 投影边属性到隐层维度
        e = self.edge_in(edge_attr)
        e = self.edge_drop(self.edge_act(e))
        e = self.edge_norm(e)

        # 多层 MPNN 更新分子和边的表示
        h = node_h
        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        # 输出标准化
        h = self.out_norm(h)
        # 计算图数量（批大小）
        num_graphs = _num_graphs_from_batch(batch, fallback=fallback_num_graphs)
        # 混合物级池化：计算每个混合物的分子特征平均
        mix_emb = global_pool_mean(h, batch, num_graphs)
        return h, mix_emb



# ----------------------------
# Token fusion transformer (for multi-scale features)
# ----------------------------
class TokenFusionTransformer(nn.Module):
    """
    令牌融合 Transformer - 多尺度特征融合
    
    使用 TransformerEncoder 对特征令牌序列进行融合和相互作用建模。该模块支持多种
    配置以适应不同的融合策略：
    
    1. **CLS 令牌**：学习的特殊令牌，通过自注意力捕捉所有输入令牌的融合表示
    2. **位置编码**：可学习的位置嵌入，为令牌序列提供位置信息
    3. **令牌类型嵌入**：可学习的令牌类型表示（如用于区分不同分子的令牌）
    4. **多头自注意力**：并行的 L 头注意力，捕捉令牌间的多种关系
    
    典型应用：融合分子图编码器的不同层或不同尺度的特征令牌。
    
    输入令牌格式：
        - tokens: 特征令牌序列，形状 (B, L, D)，B 为批大小，L 为序列长度，D 为特征维度
        - type_ids: 可选的令牌类型 ID，形状 (B, L)，范围 [0, type_vocab_size)
    
    Args:
        d_model: 令牌嵌入维度（和输入特征维度 D 必须相同）
        nhead: 多头注意力头数，默认 8
        num_layers: TransformerEncoder 层数，默认 2
        dim_feedforward: 前向网络的隐层维度，默认 1024
        dropout: 丢弃率，默认 0.10
        pool: 池化策略，"cls" 使用 CLS 令牌，"mean" 使用平均，默认 "cls"
        max_len: 位置编码的最大序列长度，默认 32
        type_vocab_size: 令牌类型的总数，默认 16
        use_type_embed: 是否使用令牌类型嵌入，默认 True
        use_pos_embed: 是否使用位置嵌入，默认 True
    
    前向传播流程：
        1. （可选）添加 CLS 令牌到序列开头
        2. 添加位置嵌入
        3. （如果提供 type_ids）添加令牌类型嵌入
        4. 通过 TransformerEncoder 处理
        5. 输出标准化
        6. 根据池化策略生成聚合表示
    
    返回:
        pooled: 聚合表示，形状 (B, D)，可用作图级预测的输入
        h: 所有令牌的最终表示，形状 (B, L 或 L+1, D)，包含 CLS 令牌（若启用）
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.10,
        pool: str = "cls",
        max_len: int = 32,
        type_vocab_size: int = 16,
        use_type_embed: bool = True,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pool = str(pool).lower()
        self.use_cls = self.pool == "cls"
        self.use_type_embed = bool(use_type_embed)
        self.use_pos_embed = bool(use_pos_embed)

        # CLS 令牌：用于汇集所有输入信息的学习参数
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        # 可学习的位置编码（固定最大长度；运行时切片）
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, int(max_len) + (1 if self.use_cls else 0), self.d_model))
        else:
            self.pos_embed = None

        # 可学习的令牌类型嵌入（用于区分不同来源的令牌）
        if self.use_type_embed:
            self.type_embed = nn.Embedding(int(type_vocab_size), self.d_model)
        else:
            self.type_embed = None

        # Transformer 编码器：多层多头自注意力
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.out_norm = nn.LayerNorm(self.d_model)

        # 参数初始化（小方差 Gaussian 初始化，符合 BERT/Vision Transformer 传统）
        nn.init.normal_(self.cls_token, std=0.02) if self.cls_token is not None else None
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)
        if self.type_embed is not None:
            nn.init.normal_(self.type_embed.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            tokens: 特征令牌序列，形状 (B, L, D)，其中 D = d_model
            type_ids: 可选的令牌类型 ID，形状 (B, L)
        
        Returns:
            pooled: 聚合表示，形状 (B, D)
            h: 处理后的令牌序列，形状 (B, L(+1 if CLS), D)
        """
        assert tokens.dim() == 3, f"tokens 必须是 (B,L,D)，得到 {tokens.shape}"
        B, L, D = tokens.shape
        assert D == self.d_model, f"令牌维度不匹配：得到 {D}，期望 {self.d_model}"
        device = tokens.device

        if type_ids is not None:
            assert type_ids.shape[:2] == (B, L), f"type_ids 必须是 (B,L)，得到 {type_ids.shape}"
            type_ids = type_ids.to(device=device, dtype=torch.long)

        # ===== 令牌序列准备 =====
        x = tokens
        # 添加 CLS 令牌到序列开头
        if self.use_cls:
            cls = self.cls_token.expand(B, 1, D).to(device=device, dtype=x.dtype)
            x = torch.cat([cls, x], dim=1)
            # 如果有类型 ID，为 CLS 令牌添加类型 0
            if type_ids is not None:
                cls_type = torch.zeros((B, 1), device=device, dtype=torch.long)
                type_ids = torch.cat([cls_type, type_ids], dim=1)

        # 添加位置嵌入
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, : x.shape[1], :].to(device=device, dtype=x.dtype)

        # 添加令牌类型嵌入
        if (self.type_embed is not None) and (type_ids is not None):
            x = x + self.type_embed(type_ids)

        # ===== Transformer 处理 =====
        # 通过多层 Transformer 编码器处理令牌
        h = self.encoder(x)
        # 输出标准化
        h = self.out_norm(h)

        # ===== 池化 =====
        if self.use_cls:
            # 使用 CLS 令牌作为聚合表示
            pooled = h[:, 0, :]
        else:
            # 或使用平均池化
            pooled = h.mean(dim=1)

        return pooled, h

# ----------------------------
# Full graph model (concat / mixture graph)
# ----------------------------

class LLEGraphNet(nn.Module):
    """
    液液平衡（LLE）图神经网络主模型 - 三元混合物预测
    
    该模型是一个通用的模块化架构，支持多种输入模式和融合策略，用于预测三元混合物的
    液液平衡曲线和相分离性质。
    
    ===== 核心特性 =====
    
    1. **分子图编码**：通过 MPNNEncoder 将每个分子的原子图转换为分子嵌入
       - 支持原子和键的迭代特征更新
       - 自动处理全局分子特征融合
    
    2. **混合物图编码**（可选）：通过 MixGraphEncoder 建模分子间相互作用
       - 在 3-节点图上运行消息传递，捕捉混合物特定的相互作用
       - 可选全局特征融合到混合物表示
    
    3. **功能基团（FG）处理**（可选）：
       - 多热编码 → 密集表示：FG 向量直接通过 MLP 投影
       - FG 令牌编码：FG ID 序列 → 嵌入 → 跨分子注意力 → 池化
       - 支持跨分子 FG 注意力，建模 FG 间的交互
    
    4. **多种融合模式**：
       - concat：直接拼接所有特征，通过 MLP 处理
       - transformer：将特征分组作为令牌，通过 TransformerEncoder 融合
       - s3_set：对称分子系统的置换等变融合
    
    5. **S3 置换等变性**（可选）：用于对称三元系统
       - 对分子进行平均"对称上下文"处理
       - 确保模型对分子置换不变
    
    ===== 输入格式 =====
    
    x_dict 包含：
        - 'g1', 'g2', 'g3': 三个分子的批处理图字典
          每个包含：x（节点特征）, edge_index, edge_attr, batch, g（全局特征）
        
        - 'scalars': 形状 (B, 2)，[T_norm, t]（归一化温度、组成）
        
        - 'mix'（可选，当 use_mix_graph=True）：批处理混合物图字典
          包含：edge_index, edge_attr, batch
        
        - 'fg' 或 'fg1','fg2','fg3'（可选，当 use_fg=True）：
          - 多热编码：形状 (B, fg_vocab_size) 或 (B, 3, fg_vocab_size)
          - 令牌编码：形状 (B, max_len) 或 (B, 3, max_len)，ID 序列（0=填充）
    
    ===== 参数 =====
    
    **分子图编码参数**:
        gnn_hidden: GNN 隐层维度，默认 256
        gnn_layers: MPNN 层数，默认 4
        dropout: MLP 丢弃率，默认 0.15
        pool: 节点池化策略，默认 "mean"
        update_edges: MPNN 是否更新边属性，默认 True
        edge_dropout: 边级丢弃率，默认 0.1
        edge_scale: 边更新残差缩放，默认 0.1
    
    **混合物图参数**:
        use_mix_graph: 是否使用混合物图，默认 False
        mix_layers: 混合物图 MPNN 层数，默认 2
        mix_edge_hidden: 混合物边投影维度，默认 64
        mix_update_edges: 是否更新混合物边属性，默认 False
        mix_edge_dropout: 混合物边丢弃率，默认 0.1
        mix_edge_scale: 混合物边残差缩放，默认 0.1
        mix_append_global: 是否融合全局特征到混合物表示，默认 True
    
    **FG 编码参数**:
        use_fg: 是否使用 FG 特征，默认 False
        fg_vocab_size: FG 词汇表大小，默认 0
        fg_hidden: FG MLP 隐层维度，默认 256
        fg_out_dim: FG 输出维度，默认 hidden（若为 None）
        fg_dropout: FG 丢弃率，默认 0.10
        fg_token_mode: 是否使用 FG 令牌模式（vs. 多热），默认 False
        fg_max_tokens: FG 令牌序列最大长度，默认 32
        fg_cross_attn: 是否启用跨分子 FG 注意力，默认 False
        fg_attn_heads: FG 注意力头数，默认 8
    
    **融合参数**:
        fusion_mode: 融合模式 ("concat"/"transformer"/"s3_set")，默认 "concat"
        tf_dim: Transformer 嵌入维度，默认 hidden
        tf_layers: Transformer 层数，默认 2
        tf_heads: Transformer 头数，默认 8
        tf_ff: Transformer 前向网络维度，默认 1024
        tf_dropout: Transformer 丢弃率，默认 0.10
        tf_pool: Transformer 池化 ("cls"/"mean")，默认 "cls"
        tf_max_len: Transformer 最大序列长度，默认 32
        tf_type_vocab: Transformer 令牌类型词汇大小，默认 16
    
    **对称性参数**:
        s3_equivariant: 是否启用 S3 置换等变性，默认 False
        use_interaction: 是否使用分子间相互作用特征，默认 True
        mlp_hidden: 最终 MLP 隐层维度，默认 512
    
    ===== 前向传播流程 =====
    
    1. **分子编码**：g1, g2, g3 → e1, e2, e3（分子嵌入）
    2. **S3 处理**（可选）：e1, e2, e3 → 应用对称上下文
    3. **混合物编码**（可选）：[e1, e2, e3] 通过混合物图 → 混合物特征
    4. **FG 编码**（可选）：fg1, fg2, fg3 → FG 表示，可选跨分子注意力
    5. **特征融合**：
       - concat: 拼接所有特征 → MLP
       - transformer: 作为令牌 → TransformerEncoder → MLP
       - s3_set: 对称加法 → 投影
    6. **输出头**：融合特征 + 标量 → MLP → [E1, E2, E3, R1, R2, R3]
       输出经 softmax 约束确保分布性质
    """
    def __init__(
        self,
        gnn_hidden: int = 256,
        gnn_layers: int = 4,
        mlp_hidden: int = 512,
        dropout: float = 0.15,
        pool: str = "mean",
        use_interaction: bool = True,
        # molecule edge update knobs (optional)
        update_edges: bool = True,
        edge_dropout: float = 0.1,
        edge_scale: float = 0.1,
        # mixture graph options (optional; keep backward compatibility)
        use_mix_graph: bool = False,
        mix_layers: int = 2,
        mix_edge_hidden: int = 64,
        mix_update_edges: bool = True,
        mix_edge_dropout: float = 0.1,
        mix_edge_scale: float = 0.1,
        mix_append_global: bool = True,
        # mix head knobs (ignored; kept for backward compatibility with train.py kwargs)
        mix_hidden: Optional[int] = None,
        mix_dropout: float = 0.10,
        # functional-group options (optional)
        use_fg: bool = True,
        fg_vocab_size: int = 0,
        fg_hidden: int = 256,
        fg_out_dim: Optional[int] = None,
        fg_dropout: float = 0.10,
        fg_token_mode: bool = True,
        fg_max_tokens: int = 32,
        fg_cross_attn: bool = True,
        fg_attn_heads: int = 8,
        # fusion options
        fusion_mode: str = "concat",
        tf_dim: Optional[int] = None,
        tf_layers: int = 2,
        tf_heads: int = 8,
        tf_ff: int = 1024,
        tf_dropout: float = 0.10,
        tf_pool: str = "cls",
        tf_max_len: int = 32,
        tf_type_vocab: int = 16,
        s3_equivariant: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden = int(gnn_hidden)
        self.use_interaction = bool(use_interaction)
        self.s3_equivariant = bool(s3_equivariant)

        self.use_mix_graph = bool(use_mix_graph)
        self.mix_append_global = bool(mix_append_global)

        self.fusion_mode = str(fusion_mode).lower().strip()
        if tf_dim is None:
            tf_dim = self.hidden
        self.tf_dim = int(tf_dim)

        # molecule encoder
        self.encoder = MPNNEncoder(
            node_dim=atom_feature_dim(),
            edge_dim=bond_feature_dim(),
            global_dim=global_feature_dim(),
            hidden=self.hidden,
            layers=int(gnn_layers),
            dropout=float(dropout),
            pool=str(pool),
            update_edges=bool(update_edges),
            edge_dropout=float(edge_dropout),
            edge_scale=float(edge_scale),
        )

        # mixture encoder (optional)
        if self.use_mix_graph:
            self.mix_encoder = MixGraphEncoder(
                hidden=self.hidden,
                layers=int(mix_layers),
                dropout=float(dropout),
                edge_hidden=int(mix_edge_hidden),
                update_edges=bool(mix_update_edges),
                edge_dropout=float(mix_edge_dropout),
                edge_scale=float(mix_edge_scale),
            )
        else:
            self.mix_encoder = None

        # FG encoder (multi-hot -> dense) or FG token encoder
        self.use_fg = bool(use_fg) and (int(fg_vocab_size) > 0)
        self.fg_vocab_size = int(fg_vocab_size)
        self.fg_out_dim = int(self.hidden if fg_out_dim is None else fg_out_dim)
        self.fg_token_mode = bool(fg_token_mode) and self.use_fg
        self.fg_max_tokens = int(fg_max_tokens)
        self.fg_cross_attn = bool(fg_cross_attn) and self.use_fg

        if self.use_fg and self.fg_token_mode:
            self.fg_token_dim = int(fg_hidden)
            self.fg_token_embed = nn.Embedding(self.fg_vocab_size + 1, self.fg_token_dim, padding_idx=0)
            self.fg_token_drop = nn.Dropout(float(fg_dropout))
            if self.fg_cross_attn:
                heads = int(fg_attn_heads)
                if heads <= 0 or (self.fg_token_dim % heads) != 0:
                    heads = 1
                self.fg_attn = nn.MultiheadAttention(
                    embed_dim=self.fg_token_dim,
                    num_heads=heads,
                    dropout=float(fg_dropout),
                    batch_first=True,
                )
                self.fg_attn_norm = nn.LayerNorm(self.fg_token_dim)
            else:
                self.fg_attn = None
                self.fg_attn_norm = None
            self.fg_token_proj = nn.Sequential(
                nn.Linear(self.fg_token_dim, self.fg_out_dim),
                nn.GELU(),
                nn.Dropout(float(fg_dropout)),
            )
            self.fg_encoder = None
        elif self.use_fg:
            self.fg_token_dim = 0
            self.fg_token_embed = None
            self.fg_token_drop = None
            self.fg_attn = None
            self.fg_attn_norm = None
            self.fg_token_proj = None
            self.fg_encoder = nn.Sequential(
                nn.Linear(self.fg_vocab_size, int(fg_hidden)),
                nn.GELU(),
                nn.Dropout(float(fg_dropout)),
                nn.Linear(int(fg_hidden), self.fg_out_dim),
                nn.GELU(),
                nn.Dropout(float(fg_dropout)),
            )
        else:
            self.fg_token_dim = 0
            self.fg_token_embed = None
            self.fg_token_drop = None
            self.fg_attn = None
            self.fg_attn_norm = None
            self.fg_token_proj = None
            self.fg_encoder = None

        # --- fusion + head ---
        if self.fusion_mode == "transformer":
            # token projections
            self.proj_mol = nn.Identity() if self.hidden == self.tf_dim else nn.Linear(self.hidden, self.tf_dim)
            self.proj_inter = nn.Identity() if self.hidden == self.tf_dim else nn.Linear(self.hidden, self.tf_dim)
            self.proj_mix = nn.Identity() if self.hidden == self.tf_dim else nn.Linear(self.hidden, self.tf_dim)
            self.proj_fg = nn.Identity() if self.fg_out_dim == self.tf_dim else nn.Linear(self.fg_out_dim, self.tf_dim)
            self.proj_scalar = nn.Linear(2, self.tf_dim)

            self.token_fuser = TokenFusionTransformer(
                d_model=self.tf_dim,
                nhead=int(tf_heads),
                num_layers=int(tf_layers),
                dim_feedforward=int(tf_ff),
                dropout=float(tf_dropout),
                pool=str(tf_pool),
                max_len=int(tf_max_len),
                type_vocab_size=int(tf_type_vocab),
                use_type_embed=True,
                use_pos_embed=True,
            )

            self.backbone = nn.Sequential(
                nn.Linear(self.tf_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.comp_backbone = None
            self.comp_head_E = None
            self.comp_head_R = None
        elif self.fusion_mode == "s3_set":
            comp_dim = self.hidden
            if self.use_interaction:
                comp_dim += 3 * self.hidden
            if self.use_mix_graph and self.mix_append_global:
                comp_dim += self.hidden
            if self.use_fg:
                comp_dim += self.fg_out_dim
            comp_dim += 2

            self.comp_backbone = nn.Sequential(
                nn.Linear(comp_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.comp_head_E = nn.Linear(mlp_hidden, 1)
            self.comp_head_R = nn.Linear(mlp_hidden, 1)

            self.backbone = None
            self.proj_mol = None
            self.proj_inter = None
            self.proj_mix = None
            self.proj_fg = None
            self.proj_scalar = None
            self.token_fuser = None
        else:
            # original concat mode
            base_dim = 3 * self.hidden
            inter_dim = 0
            if self.use_interaction:
                inter_dim = 6 * self.hidden  # abs diffs + products

            mix_dim = self.hidden if (self.use_mix_graph and self.mix_append_global) else 0
            fg_dim = 3 * self.fg_out_dim if self.use_fg else 0
            in_dim = base_dim + inter_dim + mix_dim + fg_dim + 2  # + scalars

            self.backbone = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.proj_mol = None
            self.proj_inter = None
            self.proj_mix = None
            self.proj_fg = None
            self.proj_scalar = None
            self.token_fuser = None
            self.comp_backbone = None
            self.comp_head_E = None
            self.comp_head_R = None

        self.head_E = nn.Linear(mlp_hidden, 3)
        self.head_R = nn.Linear(mlp_hidden, 3)

    def _encode_fg(self, x: Dict[str, Any], B: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        编码功能基团（FG）特征
        
        支持两种 FG 表示方式：
        1. FG 令牌模式：ID 序列 → 嵌入 → 可选跨分子注意力 → 池化
        2. FG 多热模式：多热向量 → MLP 投影
        
        Args:
            x: 输入字典，包含 FG 数据
            B: 批大小
            device: 计算设备
        
        Returns:
            (f1, f2, f3) 或 None，每个形状 (B, fg_out_dim)
        """
        if not self.use_fg:
            return None

        # ===== FG 令牌模式 =====
        if self.fg_token_mode and self.fg_token_embed is not None:
            # 获取 FG 令牌 ID
            fg_ids = x.get("fg_ids", None)
            fg1_ids = fg2_ids = fg3_ids = None
            if fg_ids is not None:
                fg_ids = fg_ids.to(device=device, dtype=torch.long)
                # 处理两种格式：(B,3,L) 或分别的 (B,L)
                if fg_ids.dim() == 3 and fg_ids.shape[1] == 3:
                    fg1_ids, fg2_ids, fg3_ids = fg_ids[:, 0, :], fg_ids[:, 1, :], fg_ids[:, 2, :]
            
            # 获取分别的 FG 令牌
            if fg1_ids is None:
                fg1_ids = x.get("fg1_ids", None)
                fg2_ids = x.get("fg2_ids", None)
                fg3_ids = x.get("fg3_ids", None)

            L = int(max(1, self.fg_max_tokens))
            # 初始化缺失的 FG 令牌为零（填充）
            if fg1_ids is None:
                fg1_ids = torch.zeros((B, L), device=device, dtype=torch.long)
            else:
                fg1_ids = fg1_ids.to(device=device, dtype=torch.long)
            if fg2_ids is None:
                fg2_ids = torch.zeros((B, L), device=device, dtype=torch.long)
            else:
                fg2_ids = fg2_ids.to(device=device, dtype=torch.long)
            if fg3_ids is None:
                fg3_ids = torch.zeros((B, L), device=device, dtype=torch.long)
            else:
                fg3_ids = fg3_ids.to(device=device, dtype=torch.long)

            # 获取或生成 FG 令牌掩码（0 表示填充，1 表示有效）
            fg1_mask = x.get("fg1_mask", None)
            fg2_mask = x.get("fg2_mask", None)
            fg3_mask = x.get("fg3_mask", None)
            if fg1_mask is None:
                fg1_mask = (fg1_ids != 0).to(dtype=torch.float32)
            else:
                fg1_mask = fg1_mask.to(device=device, dtype=torch.float32)
            if fg2_mask is None:
                fg2_mask = (fg2_ids != 0).to(dtype=torch.float32)
            else:
                fg2_mask = fg2_mask.to(device=device, dtype=torch.float32)
            if fg3_mask is None:
                fg3_mask = (fg3_ids != 0).to(dtype=torch.float32)
            else:
                fg3_mask = fg3_mask.to(device=device, dtype=torch.float32)

            # 将 FG 令牌 ID 转换为嵌入（形状 B×L×D）
            f1 = self.fg_token_drop(self.fg_token_embed(fg1_ids))
            f2 = self.fg_token_drop(self.fg_token_embed(fg2_ids))
            f3 = self.fg_token_drop(self.fg_token_embed(fg3_ids))

            # 可选跨分子 FG 注意力：建模 FG 间的相互作用
            if self.fg_cross_attn and (self.fg_attn is not None):
                # 执行三个分子间的交叉注意力
                p1, p2, p3 = cross_molecular_fg_attention(
                    f1, f2, f3,
                    mask1=fg1_mask, mask2=fg2_mask, mask3=fg3_mask,
                    attn=self.fg_attn, norm=self.fg_attn_norm, drop=self.fg_token_drop
                )
            else:
                # 直接对 FG 序列应用掩码平均池化
                p1 = _masked_mean(f1, fg1_mask)
                p2 = _masked_mean(f2, fg2_mask)
                p3 = _masked_mean(f3, fg3_mask)

            # 投影到输出维度
            f1 = self.fg_token_proj(p1)
            f2 = self.fg_token_proj(p2)
            f3 = self.fg_token_proj(p3)
            return f1, f2, f3

        # ===== FG 多热模式 =====
        if self.fg_encoder is not None:
            # 支持 "fg" (B,3,V) 或分别的 "fg1","fg2","fg3" (B,V)
            if "fg" in x and x["fg"] is not None:
                fg_all = x["fg"].to(device=device, dtype=torch.float32)
                fg1, fg2, fg3 = fg_all[:, 0, :], fg_all[:, 1, :], fg_all[:, 2, :]
            else:
                fg1 = x.get("fg1", None)
                fg2 = x.get("fg2", None)
                fg3 = x.get("fg3", None)

                # 初始化缺失的 FG 多热向量为零
                if fg1 is None:
                    fg1 = torch.zeros((B, self.fg_vocab_size), device=device, dtype=torch.float32)
                else:
                    fg1 = fg1.to(device=device, dtype=torch.float32)
                if fg2 is None:
                    fg2 = torch.zeros((B, self.fg_vocab_size), device=device, dtype=torch.float32)
                else:
                    fg2 = fg2.to(device=device, dtype=torch.float32)
                if fg3 is None:
                    fg3 = torch.zeros((B, self.fg_vocab_size), device=device, dtype=torch.float32)
                else:
                    fg3 = fg3.to(device=device, dtype=torch.float32)

            # 通过 FG 编码 MLP（多热 → fg_hidden → fg_out_dim）
            f1 = self.fg_encoder(fg1)
            f2 = self.fg_encoder(fg2)
            f3 = self.fg_encoder(fg3)
            return f1, f2, f3

        return None

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        """
        前向传播：预测三元混合物的液液平衡性质
        
        Args:
            x: 输入字典，包含 'g1','g2','g3' (分子图)，'scalars' (T_norm, t)，
               以及可选的 'mix' (混合物图)，'fg'/'fg_*' (功能基团)
        
        Returns:
            形状 (B, 6) 的张量 [E1, E2, E3, R1, R2, R3]，范围 [0,1]，
            其中 E* 和 R* 分别表示富集因子和分配比，每组和为 1
        """
        # 解包输入
        g1 = x["g1"]
        g2 = x["g2"]
        g3 = x["g3"]
        scalars = x["scalars"]  # (B,2) [T_norm, t]

        B = int(scalars.shape[0])
        device = scalars.device

        # ===== 分子图编码 =====
        # 将三个分子的原子图编码为分子嵌入
        e1 = self.encoder(g1)  # (B, hidden)
        e2 = self.encoder(g2)  # (B, hidden)
        e3 = self.encoder(g3)  # (B, hidden)

        # ===== S3 置换等变处理（可选）=====
        # 对对称分子系统应用对称上下文，确保置换不变性
        if self.s3_equivariant:
            e_stack = s3_equivariant_embedding([e1, e2, e3])  # (B, 3, hidden)
            e1, e2, e3 = e_stack[:, 0, :], e_stack[:, 1, :], e_stack[:, 2, :]

        # ===== 混合物图编码（可选）=====
        # 建模分子间的混合物特定相互作用
        mix_emb = None
        if self.use_mix_graph and (self.mix_encoder is not None):
            mix = x.get("mix", None)
            if mix is not None:
                # 堆叠三个分子嵌入为混合物图的节点特征
                node_h = torch.cat([e1, e2, e3], dim=0)  # (3*B, hidden)
                _, mix_emb = self.mix_encoder(node_h, mix, fallback_num_graphs=B)  # (B, hidden)

        # ===== 功能基团编码（可选）=====
        # 编码 FG 特征，可选跨分子注意力
        fg_tuple = self._encode_fg(x, B, device)
        if fg_tuple is not None:
            f1, f2, f3 = fg_tuple  # (B, fg_out_dim)
        else:
            f1 = f2 = f3 = None

        # ===== 融合模式 1：Transformer 融合 =====
        if self.fusion_mode == "transformer":
            # 将所有特征作为令牌序列，通过 TransformerEncoder 融合
            tokens = []
            type_ids = []

            def _append(tok: torch.Tensor, t_id: int, proj: nn.Module):
                """将令牌添加到序列，应用投影"""
                # tok: (B, d_in)
                tok = proj(tok)
                tokens.append(tok)
                type_ids.append(torch.full((B, 1), int(t_id), device=device, dtype=torch.long))

            # 令牌类型 ID（用于类型嵌入）
            T_MOL = 1    # 分子
            T_ABS = 2    # 绝对差
            T_PROD = 3   # 元素积
            T_MIX = 4    # 混合物
            T_FG = 5     # 功能基团
            T_SCALAR = 6 # 标量

            # 添加分子令牌
            _append(e1, T_MOL, self.proj_mol)
            _append(e2, T_MOL, self.proj_mol)
            _append(e3, T_MOL, self.proj_mol)

            # 添加相互作用令牌（可选）
            if self.use_interaction:
                # 绝对差：|ei - ej|
                _append(torch.abs(e1 - e2), T_ABS, self.proj_inter)
                _append(torch.abs(e1 - e3), T_ABS, self.proj_inter)
                _append(torch.abs(e2 - e3), T_ABS, self.proj_inter)
                # 元素积：ei * ej
                _append(e1 * e2, T_PROD, self.proj_inter)
                _append(e1 * e3, T_PROD, self.proj_inter)
                _append(e2 * e3, T_PROD, self.proj_inter)

            # 添加混合物令牌（可选）
            if self.use_mix_graph and self.mix_append_global:
                if mix_emb is None:
                    mix_emb = (e1 + e2 + e3) / 3.0  # 备用：平均分子嵌入
                _append(mix_emb, T_MIX, self.proj_mix)

            # 添加 FG 令牌（可选）
            if f1 is not None:
                _append(f1, T_FG, self.proj_fg)
                _append(f2, T_FG, self.proj_fg)
                _append(f3, T_FG, self.proj_fg)

            # 添加标量令牌（温度和组成）
            _append(scalars.to(device=device, dtype=torch.float32), T_SCALAR, self.proj_scalar)

            # 堆叠和融合
            tok = torch.stack(tokens, dim=1)                 # (B, L, tf_dim)
            tid = torch.cat(type_ids, dim=1)                 # (B, L)
            pooled, _ = self.token_fuser(tok, tid)           # (B, tf_dim) - Transformer 池化输出
            # MLP 处理融合表示
            h = self.backbone(pooled)  # (B, mlp_hidden)
            E = torch.softmax(self.head_E(h), dim=-1)
            R = torch.softmax(self.head_R(h), dim=-1)
            return torch.cat([E, R], dim=-1)

        # ===== 融合模式 2：S3 集合融合 =====
        if self.fusion_mode == "s3_set":
            # 为每个分子单独计算特征，考虑对称性
            if mix_emb is None and self.use_mix_graph and self.mix_append_global:
                mix_emb = (e1 + e2 + e3) / 3.0

            if self.use_interaction:
                # 计算另外两个分子的平均嵌入
                mean_23 = (e2 + e3) / 2.0
                mean_13 = (e1 + e3) / 2.0
                mean_12 = (e1 + e2) / 2.0

                def _comp_feat(ei: torch.Tensor, mean_other: torch.Tensor, fi: Optional[torch.Tensor]) -> torch.Tensor:
                    """为每个分子组装特征：自身 + 平均他者 + 相互作用"""
                    feats = [ei, mean_other, torch.abs(ei - mean_other), ei * mean_other]
                    if self.use_mix_graph and self.mix_append_global:
                        feats.append(mix_emb)
                    if fi is not None:
                        feats.append(fi)
                    feats.append(scalars.to(device=device, dtype=torch.float32))
                    return torch.cat(feats, dim=-1)

                c1 = _comp_feat(e1, mean_23, f1)
                c2 = _comp_feat(e2, mean_13, f2)
                c3 = _comp_feat(e3, mean_12, f3)
            else:
                def _comp_feat(ei: torch.Tensor, fi: Optional[torch.Tensor]) -> torch.Tensor:
                    """简单特征：自身 + 可选混合物 + 可选 FG + 标量"""
                    feats = [ei]
                    if self.use_mix_graph and self.mix_append_global:
                        feats.append(mix_emb)
                    if fi is not None:
                        feats.append(fi)
                    feats.append(scalars.to(device=device, dtype=torch.float32))
                    return torch.cat(feats, dim=-1)

                c1 = _comp_feat(e1, f1)
                c2 = _comp_feat(e2, f2)
                c3 = _comp_feat(e3, f3)

            # 堆叠三个分子的特征并通过成分特定的 MLP
            comp = torch.stack([c1, c2, c3], dim=1)  # (B, 3, feat_dim)
            h = self.comp_backbone(comp.view(B * 3, -1)).view(B, 3, -1)  # (B, 3, mlp_hidden)
            # 每个分子单独输出 E 和 R
            E_logits = self.comp_head_E(h).squeeze(-1)  # (B, 3)
            R_logits = self.comp_head_R(h).squeeze(-1)  # (B, 3)
            E = torch.softmax(E_logits, dim=1)
            R = torch.softmax(R_logits, dim=1)
            return torch.cat([E, R], dim=-1)

        # ===== 融合模式 3：拼接融合（默认）=====
        # 直接拼接所有特征并通过 MLP
        feats = [e1, e2, e3]

        # 添加相互作用特征（可选）
        if self.use_interaction:
            feats += [
                torch.abs(e1 - e2), torch.abs(e1 - e3), torch.abs(e2 - e3),  # 绝对差
                e1 * e2, e1 * e3, e2 * e3,  # 元素积
            ]

        # 添加混合物特征（可选）
        if self.use_mix_graph and self.mix_append_global:
            if mix_emb is None:
                mix_emb = (e1 + e2 + e3) / 3.0  # 备用：平均分子嵌入
            feats.append(mix_emb)

        # 添加 FG 特征（可选）
        if f1 is not None:
            feats.extend([f1, f2, f3])

        # 添加标量特征（温度、组成）
        feats.append(scalars.to(device=device, dtype=torch.float32))

        # 拼接和处理
        h = torch.cat(feats, dim=-1)
        h = self.backbone(h)  # (B, mlp_hidden)
        # 输出头：应用 softmax 约束
        E = torch.softmax(self.head_E(h), dim=-1)
        R = torch.softmax(self.head_R(h), dim=-1)
        return torch.cat([E, R], dim=-1)
