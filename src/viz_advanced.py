# -*- coding: utf-8 -*-
"""
高级可视化模块：特征重要性的多种表示形式
包含：Treemap、Rank Heatmap、Bump Chart、Beeswarm分布图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置 matplotlib 后端和风格
os.environ.setdefault("MPLBACKEND", "Agg")


def apply_publication_style():
    """应用 Nature 期刊风格"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
    })


# ============================================================================
# 1. TREEMAP - 层级重要性树图（推荐作为主图）
# ============================================================================

def plot_importance_treemap(
    importance_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    out_path: str,
    title: str = "Feature Importance Hierarchy (Treemap)",
    top_k: int = 20
) -> None:
    """
    使用 Treemap 展示多层级特征重要性聚合
    
    Args:
        importance_dict: 格式如 {
            "g1_atom": array([...]), 
            "g1_bond": array([...]),
            "g2_atom": array([...]),
            ...
        }
        feature_names_dict: 格式如 {
            "g1_atom": ["Atom C", "Atom N", ...],
            "g1_bond": ["Single Bond", ...],
            ...
        }
        out_path: 输出路径
        title: 标题
        top_k: 每类特征显示 top-k
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("⚠ plotly 未安装，跳过 Treemap")
        return
    
    # 第一层：G1/G2/G3/Mix（聚合）
    # 第二层：atom/bond/global/edge/FG
    # 第三层：具体特征
    
    labels = []
    parents = []
    values = []
    colors_list = []
    
    # 定义父层级和颜色
    molecule_groups = {}  # 聚合 G1/G2/G3
    feature_types = {}    # 聚合 atom/bond/global等
    
    # 解析 importance_dict 的键
    for key, imp_array in importance_dict.items():
        if imp_array is None or len(imp_array) == 0:
            continue
        
        # 解析 key：g1_node_feat -> mol="G1", ftype="atom"
        parts = key.split('_')
        if parts[0].startswith('g'):
            mol = parts[0].upper()  # "g1" -> "G1"
            if 'node' in key:
                ftype = 'Atom Features'
            elif 'edge' in key:
                ftype = 'Bond Features'
            else:
                ftype = 'Other'
        elif 'mix_edge' in key:
            mol = 'Mixture'
            ftype = 'Mixture-Edge Features'
        elif 'mix_node' in key:
            mol = 'Mixture'
            ftype = 'Mixture-Node Features'
        elif 'fg' in key:
            mol = 'Functional Groups'
            ftype = 'FG Features'
        else:
            mol = 'Other'
            ftype = 'Features'
        
        # 选择 top-k
        idx = np.argsort(-np.abs(imp_array))[:top_k]
        top_imp = imp_array[idx]
        
        feature_names = feature_names_dict.get(key, [f"Feat_{i}" for i in range(len(imp_array))])
        top_names = [feature_names[i] for i in idx]
        
        # 更新聚合数据
        mol_key = f"{mol} (Total)"
        if mol_key not in molecule_groups:
            molecule_groups[mol_key] = {"imp": 0, "parent": "All", "color": _get_color_for_mol(mol)}
        molecule_groups[mol_key]["imp"] += np.sum(np.abs(top_imp))
        
        ftype_key = f"{mol} - {ftype}"
        if ftype_key not in feature_types:
            feature_types[ftype_key] = {"imp": 0, "parent": mol_key, "color": _get_color_for_ftype(ftype)}
        feature_types[ftype_key]["imp"] += np.sum(np.abs(top_imp))
        
        # 添加具体特征
        for fname, imp in zip(top_names, top_imp):
            labels.append(fname)
            parents.append(ftype_key)
            values.append(np.abs(float(imp)))
            colors_list.append(_get_color_for_ftype(ftype))
    
    # 添加第二层
    for ftype_key, data in feature_types.items():
        labels.append(ftype_key)
        parents.append(data["parent"])
        values.append(data["imp"])
        colors_list.append(data["color"])
    
    # 添加第一层
    for mol_key, data in molecule_groups.items():
        labels.append(mol_key)
        parents.append(data["parent"])
        values.append(data["imp"])
        colors_list.append(data["color"])
    
    # 根节点
    labels.append("All")
    parents.append("")
    values.append(sum([data["imp"] for data in molecule_groups.values()]))
    colors_list.append("#ffffff")
    
    # 创建 Treemap
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors_list,
            colorscale='RdYlBu_r',
            cmid=0,
            line=dict(width=2, color='white')
        ),
        textposition='middle center',
        textfont=dict(size=12, family='Arial'),
        hovertemplate='<b>%{label}</b><br>Importance: %{value:.4f}<extra></extra>',
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family='Arial')),
        width=1200,
        height=800,
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(family='Arial', size=12)
    )
    
    fig.write_html(out_path.replace('.png', '.html'))
    print(f"  ✓ Treemap 已保存: {os.path.basename(out_path.replace('.png', '.html'))}")


def _get_color_for_mol(mol):
    """分子类型的颜色"""
    colors = {
        "G1": "#1f77b4",     # 蓝色
        "G2": "#ff7f0e",     # 橙色
        "G3": "#2ca02c",     # 绿色
        "Mixture": "#d62728", # 红色
        "Functional Groups": "#9467bd"  # 紫色
    }
    return colors.get(mol, "#cccccc")


def _get_color_for_ftype(ftype):
    """特征类型的颜色"""
    colors = {
        "Atom Features": "#aec7e8",
        "Bond Features": "#ffbb78",
        "Global Features": "#98df8a",
        "Mixture-Edge Features": "#ff9896",
        "Mixture-Node Features": "#c5b0d5",
        "FG Features": "#c7c7c7"
    }
    for key, color in colors.items():
        if key in ftype:
            return color
    return "#cccccc"


# ============================================================================
# 2. RANK HEATMAP - 特征排名一致性热力图
# ============================================================================

def plot_feature_rank_heatmap(
    importance_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    out_path: str,
    title: str = "Feature Rank Consistency (Cross-component)",
    top_k: int = 15,
    feature_type: str = "node"  # "node", "edge", "glob"
) -> None:
    """
    展示 G1/G2/G3 的特征排名一致性
    
    行：Top-K 特征（并集）
    列：G1/G2/G3
    颜色：排名（越深越靠前）
    
    Args:
        feature_type: "node"(原子), "edge"(键), "glob"(全局)
    """
    try:
        import seaborn as sns
    except ImportError:
        print("⚠ seaborn 未安装，跳过 Rank Heatmap")
        return
    
    apply_publication_style()
    
    # 收集所有 G1/G2/G3 的指定类型特征
    g_importance = {}
    for key, imp_array in importance_dict.items():
        if imp_array is None or len(imp_array) == 0:
            continue
        
        # 根据 feature_type 筛选
        if feature_type == "node" and 'node' not in key:
            continue
        elif feature_type == "edge" and 'edge' not in key:
            continue
        elif feature_type == "glob" and 'glob' not in key:
            continue
        
        # 排除混合物图特征
        if 'mix' in key:
            continue
        
        # 提取分子编号 (g1, g2, g3)
        mol_id = key.split('_')[0]  # "g1" from "g1_node_feat"
        
        if mol_id not in g_importance:
            g_importance[mol_id] = (imp_array, feature_names_dict.get(key, []))
    
    if len(g_importance) < 2:
        print("⚠ 数据不足，跳过 Rank Heatmap")
        return
    
    # 构建排名矩阵
    all_features = set()
    rank_data = {}
    
    for mol_id, (imp_array, feat_names) in g_importance.items():
        # 获取排名
        idx = np.argsort(-np.abs(imp_array))
        
        for rank, feat_idx in enumerate(idx):
            feat_name = feat_names[feat_idx] if feat_idx < len(feat_names) else f"Feat_{feat_idx}"
            all_features.add(feat_name)
            
            if feat_name not in rank_data:
                rank_data[feat_name] = {}
            rank_data[feat_name][mol_id] = rank + 1  # 1-indexed rank
    
    # 选择 top-k 特征（按平均排名）
    avg_ranks = {feat: np.mean(list(ranks.values())) for feat, ranks in rank_data.items()}
    top_features = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])[:top_k]
    
    # 构建 DataFrame
    rank_matrix = []
    for feat in top_features:
        row = []
        for mol_id in sorted(g_importance.keys()):
            rank = rank_data[feat].get(mol_id, np.nan)
            row.append(rank)
        rank_matrix.append(row)
    
    df_ranks = pd.DataFrame(
        rank_matrix,
        index=top_features,
        columns=sorted(g_importance.keys())
    )
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(6, max(6, len(top_features) * 0.4)))
    sns.heatmap(
        df_ranks,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',  # 越深（绿）= 排名越靠前
        cbar_kws={'label': 'Rank (Smaller Indicates Higher Importance)'},
        linewidths=1,
        linecolor='white',
        ax=ax,
        vmin=1,
        vmax=len(all_features)
    )
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Rank Heatmap 已保存: {os.path.basename(out_path)}")


def plot_combined_rank_heatmaps(
    importance_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    out_path: str,
    title: str = "Feature Rank Consistency Across Components",
    top_k: int = 12,
    color_scheme: str = "nature_green"
) -> None:
    """
    生成组合的 Rank Heatmap：一行三列（原子、化学键、全局特征）
    
    Args:
        color_scheme: "nature_green" (深绿-黄-红) 或 "nature_blue" (深蓝-白-橙红)
    """
    try:
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("⚠ seaborn 未安装，跳过组合 Rank Heatmap")
        return
    
    apply_publication_style()
    
    # Nature 级别配色方案
    if color_scheme == "nature_green":
        # 方案1：深绿-黄-红渐变（深绿=重要，红=不重要）
        colors_nature = ['#00441b', '#1b7837', '#5aae61', '#a6dba0', '#d9f0d3',
                         '#fee08b', '#fdae61', '#f46d43', '#d73027', '#a50026']
    elif color_scheme == "nature_blue":
        # 方案2：深蓝-白-橙红渐变（深蓝=重要，橙红=不重要）
        colors_nature = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                         '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
    else:
        # 默认：深紫-白-橙
        colors_nature = ['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8',
                         '#fde0dd', '#fa9fb5', '#f768a1', '#dd3497', '#7a0177']
    
    cmap_nature = LinearSegmentedColormap.from_list('nature_rank', colors_nature, N=256)
    
    # 为三种特征类型准备数据
    feature_types = [
        ("node", "Atom Features"),
        ("edge", "Bond Features"),
        ("glob", "Global Features")
    ]
    
    # 加大图形尺寸和字号
    fig, axes = plt.subplots(1, 3, figsize=(12, max(5, top_k * 0.25)))
    
    for ax_idx, (feat_type, feat_name) in enumerate(feature_types):
        ax = axes[ax_idx]
        
        # 收集该类型的 G1/G2/G3 特征
        g_importance = {}
        for key, imp_array in importance_dict.items():
            if imp_array is None or len(imp_array) == 0:
                continue
            
            # 根据 feature_type 筛选
            if feat_type == "node" and 'node' not in key:
                continue
            elif feat_type == "edge" and 'edge' not in key:
                continue
            elif feat_type == "glob" and 'glob' not in key:
                continue
            
            # 排除混合物图特征
            if 'mix' in key:
                continue
            
            mol_id = key.split('_')[0]
            if mol_id not in g_importance:
                g_importance[mol_id] = (imp_array, feature_names_dict.get(key, []))
        
        if len(g_importance) < 2:
            ax.text(0.5, 0.5, f"数据不足\n({feat_name})", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        # 构建排名矩阵
        all_features = set()
        rank_data = {}
        
        for mol_id, (imp_array, feat_names) in g_importance.items():
            idx = np.argsort(-np.abs(imp_array))
            
            for rank, feat_idx in enumerate(idx):
                feat_name_str = feat_names[feat_idx] if feat_idx < len(feat_names) else f"Feat_{feat_idx}"
                all_features.add(feat_name_str)
                
                if feat_name_str not in rank_data:
                    rank_data[feat_name_str] = {}
                rank_data[feat_name_str][mol_id] = rank + 1
        
        # 选择 top-k 特征
        avg_ranks = {feat: np.mean(list(ranks.values())) for feat, ranks in rank_data.items()}
        top_features = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])[:top_k]
        
        # 构建 DataFrame（组分标签改为 1/2/3）
        rank_matrix = []
        for feat in top_features:
            row = []
            for mol_id in sorted(g_importance.keys()):
                rank = rank_data[feat].get(mol_id, np.nan)
                row.append(rank)
            rank_matrix.append(row)
        
        # 将 g1/g2/g3 映射为 1/2/3
        component_labels = [mol_id.replace('g', '') for mol_id in sorted(g_importance.keys())]
        
        df_ranks = pd.DataFrame(
            rank_matrix,
            index=top_features,
            columns=component_labels
        )
        
        # 绘制热力图 - Nature 级别配色和字号
        sns.heatmap(
            df_ranks,
            annot=True,
            fmt='.0f',
            cmap=cmap_nature,
            cbar_kws={'label': 'Rank', 'shrink': 0.95, 'fraction': 0.2, 'pad': 0.02},
            linewidths=2,
            linecolor='white',
            ax=ax,
            vmin=1,
            vmax=len(all_features),
            annot_kws={'fontsize': 9, 'fontweight': 'bold'}  # 减小标注字号
        )
        
        # 标题和标签 - 学术级别字号
        ax.set_title(feat_name, fontsize=14, fontweight='bold', pad=8)
        ax.set_xlabel('Component', fontsize=13, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel('Feature', fontsize=13, fontweight='bold')
        else:
            ax.set_ylabel('')
        
        # 刻度字号
        ax.tick_params(axis='both', labelsize=11)
        
        # colorbar 字号 - 仅第三个子图显示标签
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        if ax_idx == 2:  # 只有第三个子图（Global Features）显示标签
            cbar.set_label('Rank (Smaller Indicates Higher Importance)', fontsize=11, fontweight='bold')
        else:
            cbar.set_label('', fontsize=11)  # 前两个子图不显示标签
        
        # 同步colorbar高度与热力图高度
        cbar.ax.set_aspect(20)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.94)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 组合 Rank Heatmap 已保存: {os.path.basename(out_path)}")


# ============================================================================
# 3. BUMP CHART - 排名流动图
# ============================================================================

def plot_bump_chart(
    importance_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    out_path: str,
    title: str = "Feature Rank Trajectories",
    top_k: int = 10
) -> None:
    """
    展示 Top-K 特征的排名如何从 G1 → G2 → G3 变化
    每条线代表一个特征的排名轨迹
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠ plotly 未安装，跳过 Bump Chart")
        return
    
    # 收集 G1/G2/G3 的排名信息
    g_importance = {}
    for key, imp_array in importance_dict.items():
        if imp_array is None or len(imp_array) == 0:
            continue
        if 'node' not in key or 'mix' in key:
            continue
        
        mol_id = key.split('_')[0]
        if mol_id not in g_importance:
            g_importance[mol_id] = (imp_array, feature_names_dict.get(key, []))
    
    if len(g_importance) < 2:
        print("⚠ 数据不足，跳过 Bump Chart")
        return
    
    # 提取排名信息
    mol_ids = sorted(g_importance.keys())
    rank_data = {}
    
    for mol_id in mol_ids:
        imp_array, feat_names = g_importance[mol_id]
        idx = np.argsort(-np.abs(imp_array))
        
        for rank, feat_idx in enumerate(idx):
            feat_name = feat_names[feat_idx] if feat_idx < len(feat_names) else f"Feat_{feat_idx}"
            if feat_name not in rank_data:
                rank_data[feat_name] = {}
            rank_data[feat_name][mol_id] = rank + 1
    
    # 选择平均排名 top-k 的特征
    avg_ranks = {feat: np.mean(list(ranks.values())) for feat, ranks in rank_data.items()}
    top_features = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])[:top_k]
    
    # 绘制 Bump chart
    fig = go.Figure()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_features)))
    
    for feat_idx, feat_name in enumerate(top_features):
        ranks = [rank_data[feat_name].get(mol_id, np.nan) for mol_id in mol_ids]
        
        fig.add_trace(go.Scatter(
            x=mol_ids,
            y=ranks,
            mode='lines+markers',
            name=feat_name,
            line=dict(width=3, color=f'rgb({int(colors[feat_idx][0]*255)},{int(colors[feat_idx][1]*255)},{int(colors[feat_idx][2]*255)})'),
            marker=dict(size=10),
            hovertemplate='<b>' + feat_name + '</b><br>Component: %{x}<br>Rank: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family='Arial')),
        xaxis_title="Component",
        yaxis_title="Rank (Smaller Indicates Higher Importance)",
        hovermode='x unified',
        width=900,
        height=600,
        font=dict(family='Arial', size=12),
        yaxis=dict(autorange='reversed')  # 反转 y 轴，使排名 1 在上方
    )
    
    fig.write_html(out_path.replace('.png', '.html'))
    print(f"  ✓ Bump Chart 已保存: {os.path.basename(out_path.replace('.png', '.html'))}")


# ============================================================================
# 4. BEESWARM - SHAP 值分布图
# ============================================================================

def plot_shap_beeswarm_distribution(
    shap_values: np.ndarray,  # (n_samples, n_features)
    feature_names: List[str],
    feature_values: Optional[np.ndarray] = None,  # (n_samples, n_features)
    out_path: str = None,
    title: str = "SHAP Value Distribution",
    top_k: int = 20
) -> None:
    """
    绘制 SHAP 值的分布，展示特征的重要性范围和离散度
    
    - 横轴：SHAP 值（有正负）
    - 纵轴：特征名
    - 颜色：特征值大小
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠ plotly 未安装，跳过 Beeswarm")
        return
    
    shap_values = np.asarray(shap_values, dtype=np.float64)
    
    # 计算全局重要性（绝对值均值）
    global_importance = np.abs(shap_values).mean(axis=0)
    
    # 选择 top-k
    top_indices = np.argsort(-global_importance)[:min(top_k, len(feature_names))]
    top_features = [feature_names[i] for i in top_indices]
    
    fig = go.Figure()
    
    # 限制样本数以提高性能
    n_samples_show = min(500, shap_values.shape[0])
    
    for feat_idx, feat_id in enumerate(top_indices):
        shap_vals = shap_values[:n_samples_show, feat_id]
        
        # 特征值用于着色（如果有）
        if feature_values is not None:
            feat_vals = feature_values[:n_samples_show, feat_id]
            feat_vals_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-10)
            colors = feat_vals_norm
            colorscale = 'Viridis'
        else:
            colors = None
            colorscale = None
        
        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=[feat_id] * len(shap_vals),
            mode='markers',
            name=feature_names[feat_id],
            marker=dict(
                size=6,
                color=colors,
                colorscale=colorscale,
                showscale=(feat_idx == 0 and feature_values is not None),
                colorbar=dict(title="Feature Value") if feat_idx == 0 and feature_values is not None else None,
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            text=[f"SHAP: {v:.4f}" for v in shap_vals],
            hovertemplate='<b>' + feature_names[feat_id] + '</b><br>%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family='Arial')),
        xaxis_title="SHAP value (impact on output)",
        yaxis_title="Feature",
        height=max(600, len(top_features) * 30),
        width=1000,
        hovermode='closest',
        font=dict(family='Arial', size=11),
        yaxis=dict(
            ticktext=top_features,
            tickvals=top_indices,
            autorange='reversed'
        ),
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )
    
    if out_path:
        fig.write_html(out_path.replace('.png', '.html'))
        print(f"  ✓ Beeswarm 分布图已保存: {os.path.basename(out_path.replace('.png', '.html'))}")
    else:
        fig.show()


# ============================================================================
# 5. IMPORTANCE SUMMARY - 多子图综合展示
# ============================================================================

def plot_importance_summary(
    importance_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    shap_values_dict: Optional[Dict[str, np.ndarray]] = None,
    out_dir: str = None,
    prefix: str = "summary"
) -> None:
    """
    生成所有高级可视化图表
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "./results"
    
    print("\n📊 生成高级可视化图表...")
    
    # 1. Treemap（主图）
    try:
        treemap_path = os.path.join(out_dir, f"{prefix}_importance_treemap.png")
        plot_importance_treemap(importance_dict, feature_names_dict, treemap_path, top_k=15)
    except Exception as e:
        print(f"⚠ Treemap 生成失败: {e}")
    
    # 2. Rank Heatmap（生成两种配色方案的组合图）
    for color_scheme, suffix in [("nature_green", "green"), ("nature_blue", "blue")]:
        try:
            combined_heatmap_path = os.path.join(out_dir, f"{prefix}_rank_heatmap_combined_{suffix}.png")
            plot_combined_rank_heatmaps(
                importance_dict, 
                feature_names_dict, 
                combined_heatmap_path,
                title="Feature Rank Consistency Across Components",
                top_k=12,
                color_scheme=color_scheme
            )
        except Exception as e:
            print(f"⚠ 组合 Rank Heatmap ({suffix}) 生成失败: {e}")
    
    # 2b. 也生成独立的三张图（可选）
    for feat_type, feat_name in [("node", "Atom"), ("edge", "Bond"), ("glob", "Global")]:
        try:
            heatmap_path = os.path.join(out_dir, f"{prefix}_rank_heatmap_{feat_name.lower()}.png")
            plot_feature_rank_heatmap(
                importance_dict, 
                feature_names_dict, 
                heatmap_path, 
                title=f"{feat_name} Feature Rank Consistency",
                top_k=12,
                feature_type=feat_type
            )
        except Exception as e:
            print(f"⚠ {feat_name} Rank Heatmap 生成失败: {e}")
    
    # 3. Bump Chart
    try:
        bump_path = os.path.join(out_dir, f"{prefix}_bump_chart.html")
        plot_bump_chart(importance_dict, feature_names_dict, bump_path, top_k=8)
    except Exception as e:
        print(f"⚠ Bump Chart 生成失败: {e}")
    
    # 4. Beeswarm（如果有 SHAP 值）
    if shap_values_dict:
        for key, shap_vals in shap_values_dict.items():
            try:
                feat_names = feature_names_dict.get(key, [])
                beeswarm_path = os.path.join(out_dir, f"{prefix}_beeswarm_{key}.html")
                plot_shap_beeswarm_distribution(
                    shap_vals['importance'] if isinstance(shap_vals, dict) else shap_vals,
                    feat_names,
                    feature_values=shap_vals.get('values') if isinstance(shap_vals, dict) else None,
                    out_path=beeswarm_path,
                    title=f"SHAP Distribution: {key}",
                    top_k=15
                )
            except Exception as e:
                print(f"⚠ Beeswarm ({key}) 生成失败: {e}")
    
    print("✅ 高级可视化图表生成完成！")
