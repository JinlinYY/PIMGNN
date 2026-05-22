# web_backend/utils/plot_utils.py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import base64
from io import BytesIO
from typing import List, Tuple

def generate_ternary_plot(smiles_list: List[str], temperature: float, e_compositions: List[float], r_compositions: List[float]) -> str:
    """生成三元相图并返回base64编码"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 三元相图坐标转换
    def barycentric_to_cartesian(a, b, c):
        x = 0.5 * (2 * b + c) / (a + b + c)
        y = (np.sqrt(3) / 2) * c / (a + b + c)
        return x, y
    
    # 假设生成一些虚拟的tie-lines数据（实际应从模型预测）
    # 这里简化，生成一个简单的tie-line
    e1, e2, e3 = e_compositions
    r1, r2, r3 = r_compositions
    
    # 绘制E相点（红色）
    ex, ey = barycentric_to_cartesian(e1, e2, e3)
    ax.scatter(ex, ey, color='red', s=100, label='Extract Phase (E)')
    
    # 绘制R相点（蓝色）
    rx, ry = barycentric_to_cartesian(r1, r2, r3)
    ax.scatter(rx, ry, color='blue', s=100, label='Raffinate Phase (R)')
    
    # 绘制tie-line
    ax.plot([ex, rx], [ey, ry], 'k--', linewidth=2)
    
    # 设置轴标签
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.sqrt(3)/2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 添加顶点标签
    ax.text(0.5, np.sqrt(3)/2 + 0.05, f'Component 1\n{smiles_list[0]}', ha='center')
    ax.text(0, -0.05, f'Component 2\n{smiles_list[1]}', ha='center')
    ax.text(1, -0.05, f'Component 3\n{smiles_list[2]}', ha='center')
    
    # 添加温度信息
    ax.text(0.5, 0.1, f'Temperature: {temperature} K', ha='center', fontsize=12)
    
    ax.legend()
    
    # 转换为base64
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return f"data:image/png;base64,{img_base64}"