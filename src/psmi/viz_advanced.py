# -*- coding: utf-8 -*-
"""Create advanced publication figures for PSMI explanation results."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


os.environ.setdefault("MPLBACKEND", "Agg")


def apply_publication_style():
    """Apply publication style."""
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



def plot_importance_treemap(
    importance_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    out_path: str,
    title: str = "Feature Importance Hierarchy (Treemap)",
    top_k: int = 20
) -> None:
    """Args:
            "g1_atom": array([...]),
            "g1_bond": array([...]),
            "g2_atom": array([...]),
            ...
        }
            "g1_atom": ["Atom C", "Atom N", ...],
            "g1_bond": ["Single Bond", ...],
            ...
        }
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("[WARN] plotly Not Installed , skip Treemap")
        return
    
    
    
    
    
    labels = []
    parents = []
    values = []
    colors_list = []
    
    
    molecule_groups = {}  
    feature_types = {}    
    
    
    for key, imp_array in importance_dict.items():
        if imp_array is None or len(imp_array) == 0:
            continue
        
        
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
        
        
        idx = np.argsort(-np.abs(imp_array))[:top_k]
        top_imp = imp_array[idx]
        
        feature_names = feature_names_dict.get(key, [f"Feat_{i}" for i in range(len(imp_array))])
        top_names = [feature_names[i] for i in idx]
        
        
        mol_key = f"{mol} (Total)"
        if mol_key not in molecule_groups:
            molecule_groups[mol_key] = {"imp": 0, "parent": "All", "color": _get_color_for_mol(mol)}
        molecule_groups[mol_key]["imp"] += np.sum(np.abs(top_imp))
        
        ftype_key = f"{mol} - {ftype}"
        if ftype_key not in feature_types:
            feature_types[ftype_key] = {"imp": 0, "parent": mol_key, "color": _get_color_for_ftype(ftype)}
        feature_types[ftype_key]["imp"] += np.sum(np.abs(top_imp))
        
        
        for fname, imp in zip(top_names, top_imp):
            labels.append(fname)
            parents.append(ftype_key)
            values.append(np.abs(float(imp)))
            colors_list.append(_get_color_for_ftype(ftype))
    
    
    for ftype_key, data in feature_types.items():
        labels.append(ftype_key)
        parents.append(data["parent"])
        values.append(data["imp"])
        colors_list.append(data["color"])
    
    
    for mol_key, data in molecule_groups.items():
        labels.append(mol_key)
        parents.append(data["parent"])
        values.append(data["imp"])
        colors_list.append(data["color"])
    
    
    labels.append("All")
    parents.append("")
    values.append(sum([data["imp"] for data in molecule_groups.values()]))
    colors_list.append("#ffffff")
    
    
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
    print(f" [OK] Treemap saved : {os.path.basename(out_path.replace('.png', '.html'))}")


def _get_color_for_mol(mol):
    """Return color for mol."""
    colors = {
        "G1": "#1f77b4",     
        "G2": "#ff7f0e",     
        "G3": "#2ca02c",     
        "Mixture": "#d62728", 
        "Functional Groups": "#9467bd"  
    }
    return colors.get(mol, "#cccccc")


def _get_color_for_ftype(ftype):
    """Return color for ftype."""
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



def plot_feature_rank_heatmap(
    importance_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    out_path: str,
    title: str = "Feature Rank Consistency (Cross-component)",
    top_k: int = 15,
    feature_type: str = "node"  # "node", "edge", "glob"
) -> None:
    """Plot feature rank heatmap."""
    try:
        import seaborn as sns
    except ImportError:
        print("[WARN] seaborn Not Installed , skip Rank Heatmap")
        return
    
    apply_publication_style()
    
    
    g_importance = {}
    for key, imp_array in importance_dict.items():
        if imp_array is None or len(imp_array) == 0:
            continue
        
        
        if feature_type == "node" and 'node' not in key:
            continue
        elif feature_type == "edge" and 'edge' not in key:
            continue
        elif feature_type == "glob" and 'glob' not in key:
            continue
        
        
        if 'mix' in key:
            continue
        
        
        mol_id = key.split('_')[0]  # "g1" from "g1_node_feat"
        
        if mol_id not in g_importance:
            g_importance[mol_id] = (imp_array, feature_names_dict.get(key, []))
    
    if len(g_importance) < 2:
        print("[WARN] insufficient data , skip Rank Heatmap")
        return
    
    
    all_features = set()
    rank_data = {}
    
    for mol_id, (imp_array, feat_names) in g_importance.items():
        
        idx = np.argsort(-np.abs(imp_array))
        
        for rank, feat_idx in enumerate(idx):
            feat_name = feat_names[feat_idx] if feat_idx < len(feat_names) else f"Feat_{feat_idx}"
            all_features.add(feat_name)
            
            if feat_name not in rank_data:
                rank_data[feat_name] = {}
            rank_data[feat_name][mol_id] = rank + 1  # 1-indexed rank
    
    
    avg_ranks = {feat: np.mean(list(ranks.values())) for feat, ranks in rank_data.items()}
    top_features = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])[:top_k]
    
    
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
    
    
    fig, ax = plt.subplots(figsize=(6, max(6, len(top_features) * 0.4)))
    sns.heatmap(
        df_ranks,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',  
        cbar_kws={'label': 'Rank (Smaller Indicates Higher Importance)'},
        linewidths=1,
        linecolor='white',
        ax=ax,
        vmin=1,
        vmax=len(all_features)
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [OK] Rank Heatmap saved : {os.path.basename(out_path)}")


def plot_combined_rank_heatmaps(
    importance_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    out_path: str,
    title: str = "Feature Rank Consistency Across Components",
    top_k: int = 12,
    color_scheme: str = "nature_green",
    font_scale: float = 1.0
) -> None:
    """Plot combined rank heatmaps."""
    try:
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("[WARN] seaborn Not Installed , skip combination Rank Heatmap")
        return
    
    apply_publication_style()
    
    
    if color_scheme == "nature_green":
        
        colors_nature = ['#00441b', '#1b7837', '#5aae61', '#a6dba0', '#d9f0d3',
                         '#fee08b', '#fdae61', '#f46d43', '#d73027', '#a50026']
    elif color_scheme == "nature_blue":
        
        colors_nature = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                         '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
    else:
        
        colors_nature = ['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8',
                         '#fde0dd', '#fa9fb5', '#f768a1', '#dd3497', '#7a0177']
    
    cmap_nature = LinearSegmentedColormap.from_list('nature_rank', colors_nature, N=256)
    
    
    feature_types = [
        ("node", "Atom Features"),
        ("edge", "Bond Features"),
        ("glob", "Global Features")
    ]
    
    
    fig, axes = plt.subplots(1, 3, figsize=(14, max(3.5, top_k * 0.18)))
    
    for ax_idx, (feat_type, feat_name) in enumerate(feature_types):
        ax = axes[ax_idx]
        
        
        g_importance = {}
        for key, imp_array in importance_dict.items():
            if imp_array is None or len(imp_array) == 0:
                continue
            
            
            if feat_type == "node" and 'node' not in key:
                continue
            elif feat_type == "edge" and 'edge' not in key:
                continue
            elif feat_type == "glob" and 'glob' not in key:
                continue
            
            
            if 'mix' in key:
                continue
            
            mol_id = key.split('_')[0]
            if mol_id not in g_importance:
                g_importance[mol_id] = (imp_array, feature_names_dict.get(key, []))
        
        if len(g_importance) < 2:
            ax.text(0.5, 0.5, f" insufficient data \n({feat_name})", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        
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
        
        
        avg_ranks = {feat: np.mean(list(ranks.values())) for feat, ranks in rank_data.items()}
        top_features = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])[:top_k]
        
        
        rank_matrix = []
        for feat in top_features:
            row = []
            for mol_id in sorted(g_importance.keys()):
                rank = rank_data[feat].get(mol_id, np.nan)
                row.append(rank)
            rank_matrix.append(row)
        
        
        component_labels = [mol_id.replace('g', '') for mol_id in sorted(g_importance.keys())]
        
        df_ranks = pd.DataFrame(
            rank_matrix,
            index=top_features,
            columns=component_labels
        )
        
        
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
            annot_kws={'fontsize': 9, 'fontweight': 'bold'}  
        )
        
        
        ax.set_title(feat_name, fontsize=int(12*font_scale), pad=8)
        ax.set_xlabel('Component', fontsize=int(12*font_scale))
        if ax_idx == 0:
            ax.set_ylabel('Feature', fontsize=int(12*font_scale))
        else:
            ax.set_ylabel('')
        
        
        ax.tick_params(axis='both', labelsize=int(12*font_scale))
        
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=int(12*font_scale))
        if ax_idx == 2:  
            cbar.set_label('Smaller Indicates Higher Importance', fontsize=int(12*font_scale))
        else:
            cbar.set_label('', fontsize=int(12*font_scale))  
        
        
        cbar.ax.set_aspect(20)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [OK] combination Rank Heatmap saved : {os.path.basename(out_path)}")



def plot_bump_chart(
    importance_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    out_path: str,
    title: str = "Feature Rank Trajectories",
    top_k: int = 10
) -> None:
    """Plot bump chart."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly Not Installed , skip Bump Chart")
        return
    
    
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
        print("[WARN] insufficient data , skip Bump Chart")
        return
    
    
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
    
    
    avg_ranks = {feat: np.mean(list(ranks.values())) for feat, ranks in rank_data.items()}
    top_features = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])[:top_k]
    
    
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
        yaxis=dict(autorange='reversed')  
    )
    
    fig.write_html(out_path.replace('.png', '.html'))
    print(f" [OK] Bump Chart saved : {os.path.basename(out_path.replace('.png', '.html'))}")



def plot_shap_beeswarm_distribution(
    shap_values: np.ndarray,  # (n_samples, n_features)
    feature_names: List[str],
    feature_values: Optional[np.ndarray] = None,  # (n_samples, n_features)
    out_path: str = None,
    title: str = "SHAP Value Distribution",
    top_k: int = 20
) -> None:
    """Plot shap beeswarm distribution."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARN] plotly Not Installed , skip Beeswarm")
        return
    
    shap_values = np.asarray(shap_values, dtype=np.float64)
    
    
    global_importance = np.abs(shap_values).mean(axis=0)
    
    
    top_indices = np.argsort(-global_importance)[:min(top_k, len(feature_names))]
    top_features = [feature_names[i] for i in top_indices]
    
    fig = go.Figure()
    
    
    n_samples_show = min(500, shap_values.shape[0])
    
    for feat_idx, feat_id in enumerate(top_indices):
        shap_vals = shap_values[:n_samples_show, feat_id]
        
        
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
        print(f" [OK] Beeswarm distribution plot saved : {os.path.basename(out_path.replace('.png', '.html'))}")
    else:
        fig.show()



def plot_importance_summary(
    importance_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]],
    shap_values_dict: Optional[Dict[str, np.ndarray]] = None,
    out_dir: str = None,
    prefix: str = "summary",
    font_scale: float = 1.0
) -> None:
    """Plot importance summary."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = str(
            Path(__file__).resolve().parents[2]
            / "figures"
            / "08_interpretability"
            / "reproduction"
        )
    
    print("\n📊 generate advanced visualizations ...")
    
    
    try:
        treemap_path = os.path.join(out_dir, f"{prefix}_importance_treemap.png")
        plot_importance_treemap(importance_dict, feature_names_dict, treemap_path, top_k=15)
    except Exception as e:
        print(f"[WARN] Treemap generation failed : {e}")
    
    
    for color_scheme, suffix in [("nature_green", "green"), ("nature_blue", "blue")]:
        try:
            combined_heatmap_path = os.path.join(out_dir, f"{prefix}_rank_heatmap_combined_{suffix}.png")
            plot_combined_rank_heatmaps(
                importance_dict, 
                feature_names_dict, 
                combined_heatmap_path,
                title="Feature Rank Consistency Across Components",
                top_k=12,
                color_scheme=color_scheme,
                font_scale=font_scale
            )
        except Exception as e:
            print(f"[WARN] combination Rank Heatmap ({suffix}) generation failed : {e}")
    
    
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
            print(f"[WARN] {feat_name} Rank Heatmap generation failed : {e}")
    
    # 3. Bump Chart
    try:
        bump_path = os.path.join(out_dir, f"{prefix}_bump_chart.html")
        plot_bump_chart(importance_dict, feature_names_dict, bump_path, top_k=8)
    except Exception as e:
        print(f"[WARN] Bump Chart generation failed : {e}")
    
    
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
                print(f"[WARN] Beeswarm ({key}) generation failed : {e}")
    
    print("[OK] advanced visualization complete !")
