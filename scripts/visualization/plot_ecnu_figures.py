"""Create publication figures for ECNU application-case comparisons."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
from pathlib import Path



file_name = '应用案例-all_merged.xlsx' 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
file_path = PROJECT_ROOT / "datasets" / "processed" / "应用案例-all_merged.xlsx"

print(f"正在尝试读取文件: {file_path}")


try:
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        
        
        df = pd.read_excel(file_path)
    else:
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            
            print("UTF-8 读取失败，尝试 GBK 编码...")
            df = pd.read_csv(file_path, encoding='gbk')

    print("[OK] 文件读取成功！")
    print(f"包含列: {df.columns.tolist()}")

except FileNotFoundError:
    print(f"[ERROR] 找不到文件：{file_path}")
    print(f"系统寻找的路径是: {file_path}")
    exit()
except Exception as e:
    print(f"[ERROR] 读取出错: {e}")
    print("建议：如果是 .xlsx 文件，请确保安装了依赖库: pip install openpyxl")
    exit()




if 'Model' in df.columns:
    df['Model'] = df['Model'].replace('COSMO-rs', 'COSMO-RS')


plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


model_colors = {
    'Experiment': 'black',
    'PSMI': '#E64B35',   
    'COSMO-RS': '#4DBBD5', # NPG Blue
    'NRTL': '#00A087',     # NPG Green
    'UNIFAC': '#3C5488'    # NPG Dark Blue
}


phase_markers = {
    'Extract': 'o',
    'Raffinate': '^'
}


def to_ternary_rotated(x1, x2, x3):
    # x1: Solvent, x2: Benzene, x3: Hexane
    x = x2 + 0.5 * x3
    y = (np.sqrt(3)/2) * x3
    return x, y


df['_c1_norm'] = df['Component 1'].astype(str).str.strip().str.lower()
df['_c2_norm'] = df['Component 2'].astype(str).str.strip().str.lower()
df['_c3_norm'] = df['Component 3'].astype(str).str.strip().str.lower()


systems = (
    df[['LLE system NO.', '_c1_norm', '_c2_norm', '_c3_norm']]
    .drop_duplicates()
    .to_dict('records')
)

output_dir = PROJECT_ROOT / "figures" / "09_application_cases" / "nature_plots_reproduction"
os.makedirs(output_dir, exist_ok=True)

for sys in systems:
    
    sys_df = df[
        (df['_c1_norm'] == sys['_c1_norm']) &
        (df['_c2_norm'] == sys['_c2_norm']) &
        (df['_c3_norm'] == sys['_c3_norm'])
    ]
    
    if sys_df.empty:
        print("跳过体系 (无数据)")
        continue

    c1 = sys_df['Component 1'].iloc[0]
    c2 = sys_df['Component 2'].iloc[0]
    c3 = sys_df['Component 3'].iloc[0]
        
    temps = sorted(sys_df['T/K'].unique())
    temps = temps[:3]

    if len(temps) == 0:
        continue

    fig, axes = plt.subplots(1, len(temps), figsize=(5 * len(temps), 4.5), sharex=True, sharey=True)
    if len(temps) == 1:
        axes = [axes]

    
    models_order = ['Experiment', 'PSMI', 'COSMO-RS', 'NRTL', 'UNIFAC']

    for ax, temp in zip(axes, temps):
        t_df = sys_df[sys_df['T/K'] == temp]

        
        A = [0, 0]
        B = [0.5, np.sqrt(3)/2]
        C = [1, 0]

        ax.plot([A[0], C[0]], [A[1], C[1]], 'k-', lw=1.0)
        ax.plot([C[0], B[0]], [C[1], B[1]], 'k-', lw=1.0)
        ax.plot([B[0], A[0]], [B[1], A[1]], 'k-', lw=1.0)

        
        offset = 0.05
        if ax is axes[0]:
            ax.text(A[0]-offset, A[1]-0.02, c1, ha='right', va='center', fontsize=10)
            ax.text(C[0]+offset, C[1]-0.02, c2, ha='left', va='center', fontsize=10)
            ax.text(B[0], B[1]+offset, c3, ha='center', va='bottom', fontsize=10)

        for model in models_order:
            m_df = t_df[t_df['Model'] == model]
            if m_df.empty:
                continue

            color = model_colors.get(model, 'gray')

            ex_x = []
            ex_y = []
            for _, row in m_df.iterrows():
                x, y = to_ternary_rotated(row['Ex1'], row['Ex2'], row['Ex3'])
                ex_x.append(x)
                ex_y.append(y)

            ax.scatter(ex_x, ex_y, color=color, marker=phase_markers['Extract'],
                       s=36, alpha=0.9, edgecolors='none')

            rx_x = []
            rx_y = []
            for _, row in m_df.iterrows():
                x, y = to_ternary_rotated(row['Rx1'], row['Rx2'], row['Rx3'])
                rx_x.append(x)
                rx_y.append(y)

            ax.scatter(rx_x, rx_y, color=color, marker=phase_markers['Raffinate'],
                       s=36, alpha=0.9, edgecolors='none')

            
            for (x1, y1), (x2, y2) in zip(zip(ex_x, ex_y), zip(rx_x, rx_y)):
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.35, linewidth=0.8)

        ax.axis('off')
        ax.set_aspect('equal')
        ax.text(0.5, 1.05, f"{temp} K",
            ha='center', va='bottom', transform=ax.transAxes, fontsize=12)

    
    legend_handles = []
    for model in models_order:
        if model not in sys_df['Model'].unique():
            continue
        color = model_colors.get(model, 'gray')
        legend_handles.append(
            mlines.Line2D([], [], color=color, marker=phase_markers['Extract'], linestyle='None',
                          markersize=8, label=f"{model} Extract")
        )
        legend_handles.append(
            mlines.Line2D([], [], color=color, marker=phase_markers['Raffinate'], linestyle='None',
                          markersize=8, label=f"{model} Raffinate")
        )

    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.02, 0.98),
               frameon=False, fontsize=10)

    fig.suptitle(f"{c1} + {c2} + {c3}", fontsize=12)

    
    sys_no = sys_df['LLE system NO.'].iloc[0] if 'LLE system NO.' in sys_df.columns else 'NA'
    filename = f"system{sys_no}_{c1}_row.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 已保存图片: {save_path}")

print("所有图片绘制完成！")
