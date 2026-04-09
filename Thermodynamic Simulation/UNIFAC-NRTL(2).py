import pandas as pd
import numpy as np
from phasepy import component, mixture
from phasepy import virialgamma 
from phasepy.equilibrium import lle

# ==========================================
# 1. 定义纯组分
# ==========================================
# 体系2组分：正庚烷 + 甲苯 + 环丁砜
n_heptane = component(name='n-Heptane', Tc=540.2, Pc=27.4, Zc=0.261, Vc=432.0, w=0.349, GC={'CH3':2, 'CH2':5})
toluene = component(name='Toluene', Tc=591.8, Pc=41.0, Zc=0.264, Vc=316.0, w=0.263, GC={'ACH':5, 'ACCH3':1})
sulfolane = component(name='Sulfolane', Tc=800.0, Pc=45.0, Zc=0.25, Vc=285.0, w=0.6, GC={'CH2':4, 'TMSO2':1})

# ==========================================
# 2. 构造“借壳上市” UNIFAC 注入器
# ==========================================
def build_custom_unifac_model(mix):
    """利用合法体系骗过初始化检查，然后将内部矩阵替换为自定义矩阵"""
    # 使用标准基团建立一个假的混合物以通过验证
    dummy_comp1 = component(name='dummy1', Tc=600, Pc=30, w=0.3, GC={'CH3':1, 'CH2':1})
    dummy_comp2 = component(name='dummy2', Tc=600, Pc=30, w=0.3, GC={'ACH':6})
    dummy_comp3 = component(name='dummy3', Tc=600, Pc=30, w=0.3, GC={'CH2':2})
    dummy_mix = dummy_comp1 + dummy_comp2 + dummy_comp3
    dummy_mix.unifac()
    model = virialgamma(dummy_mix, virialmodel='ideal_gas', actmodel='unifac')
    
    # 填入体系2所需的基团参数 (包含新加入的芳香烃基团 ACH, ACCH3)
    Rk_db = {'CH3': 0.9011, 'CH2': 0.6744, 'ACH': 0.5313, 'ACCH3': 1.2663, 'TMSO2': 1.6}
    Qk_db = {'CH3': 0.848,  'CH2': 0.540,  'ACH': 0.400,  'ACCH3': 0.968,  'TMSO2': 1.4}
    
    # 主基团映射：烷烃=1, 芳香烃=3, 环丁砜=40
    mg_map = {'CH3': 1, 'CH2': 1, 'ACH': 3, 'ACCH3': 3, 'TMSO2': 40}
    
    # a0_db 交互参数矩阵 (占位符参数，如果你有针对芳烃和环丁砜回归的专用参数，请修改主基团3和40的交互值)
    a0_db = {
        1:  {1: 0.0,    3: 61.13,  40: 1250.0},
        3:  {1: -11.12, 3: 0.0,    40: 300.0},  # 芳烃与环丁砜(预估值)
        40: {1: 300.0,  3: 50.0,   40: 0.0}     # 环丁砜与芳烃(预估值)
    }
    
    unique_groups = []
    for gc_dict in mix.GC:
        if gc_dict:
            for g in gc_dict.keys():
                if g not in unique_groups:
                    unique_groups.append(g)
                    
    ng = len(unique_groups)
    nc = mix.nc
    
    Vk = np.zeros((nc, ng))
    Qk = np.zeros(ng)
    Rk = np.zeros(ng)
    
    for i, gc_dict in enumerate(mix.GC):
        if gc_dict:
            for g, count in gc_dict.items():
                idx = unique_groups.index(g)
                Vk[i, idx] = count
                Qk[idx] = Qk_db[g]
                Rk[idx] = Rk_db[g]
                
    a0 = np.zeros((ng, ng))
    for i, g1 in enumerate(unique_groups):
        for j, g2 in enumerate(unique_groups):
            a0[i, j] = a0_db.get(mg_map[g1], {}).get(mg_map[g2], 0.0)
            
    qi = np.sum(Vk * Qk, axis=1)
    ri = np.sum(Vk * Rk, axis=1)

    model.mix = mix
    model.nc = nc
    
    model.Vk = np.ascontiguousarray(Vk, dtype=np.float64)
    model.Qk = np.ascontiguousarray(Qk, dtype=np.float64)
    model.a0 = np.ascontiguousarray(a0, dtype=np.float64)
    
    zeros = np.ascontiguousarray(np.zeros_like(a0), dtype=np.float64)
    model.a1 = zeros; model.a2 = zeros; model.a3 = zeros
    model.b0 = zeros; model.b1 = zeros; model.b2 = zeros
    model.c0 = zeros; model.c1 = zeros; model.c2 = zeros
    
    model.q = np.ascontiguousarray(qi, dtype=np.float64)
    model.r = np.ascontiguousarray(ri, dtype=np.float64)
    
    return model

# ==========================================
# 3. 全区间相平衡数据生成器
# ==========================================
def generate_broad_phase_envelope(sys_no, model_name, model, T, P, solute_idx, x0_guess, w0_guess, max_solute=0.7, n_points=80):
    """
    通过扫描溶质组成，获取尽可能宽阔的两相区包络线
    """
    print(f"\n--- 正在为 体系 {sys_no} ({model_name}) 生成宽区间相图数据 (T={T}K) ---")
    
    results = []
    x0 = np.array(x0_guess)
    w0 = np.array(w0_guess)
    
    # 溶质比例从极少量 (0.1%) 逐渐增加到 max_solute
    solute_fractions = np.linspace(0.001, max_solute, n_points)
    
    for solute_z in solute_fractions:
        # 构造全局进料组成 z (假设另外两组分按 1:1 混合，这足以扫出完整的双节线)
        z = np.zeros(3)
        z[solute_idx] = solute_z
        remains = (1.0 - solute_z) / 2.0
        for i in range(3):
            if i != solute_idx:
                z[i] = remains
                
        try:
            res = lle(x0, w0, z, T, P, model)
            x_res, w_res = (res[0], res[1]) if isinstance(res, tuple) else (res.x, res.w)
            
            # 记录数据：x为富萃取剂相，w为富稀释剂相
            results.append({
                'Feed_Solute': np.round(solute_z, 4),
                'Ex1': np.round(x_res[0], 4), 'Ex2': np.round(x_res[1], 4), 'Ex3': np.round(x_res[2], 4),
                'Rx1': np.round(w_res[0], 4), 'Rx2': np.round(w_res[1], 4), 'Rx3': np.round(w_res[2], 4)
            })
            
            # 连续延拓法的核心：用上一个收敛的结果作为下一个点的初值猜测
            x0, w0 = x_res, w_res 
            
        except Exception as e:
            print(f"  > 溶质进料达 {solute_z:.2f} 时未收敛或越界，已到达临界互溶点 (Plait Point) 附近。")
            break
            
    if results:
        df_dense = pd.DataFrame(results)
        output_name = f'Sys{sys_no}_{T}K_{model_name}_宽区间相图.xlsx'
        df_dense.to_excel(output_name, index=False)
        print(f"✅ 体系 {sys_no} ({model_name}) 生成完毕！共获取 {len(df_dense)} 个平衡点，已保存至 {output_name}")
    else:
        print(f"❌ 体系 {sys_no} ({model_name}) 生成失败，请检查初始猜测值。")

# ==========================================
# 4. 配置模型并运行批量仿真
# ==========================================
a_default = np.array([[0., 0.2, 0.2], [0.2, 0., 0.2], [0.2, 0.2, 0.]])

# NRTL 交互参数占位矩阵 (你需要替换为回归好的针对正庚烷-甲苯-环丁砜体系的参数)
g_sys2 = np.array([[0., 100., 1200.], [-20., 0., 50.], [1000., 30., 0.]]) 

# ----------------- 体系 2: 正庚烷(0) + 甲苯(1) + 环丁砜(2) -----------------
mix2 = n_heptane + toluene + sulfolane
mod2_uni = build_custom_unifac_model(mix2)
mix2.NRTL(a_default, g_sys2, np.zeros_like(g_sys2))
mod2_nrtl = virialgamma(mix2, virialmodel='ideal_gas', actmodel='nrtl')

# 根据你的 CSV 数据文件，设定体系2特定温度和压力
temps_sys2 = [348.15] # 提取自源数据文件 
P_sys2 = 1.013 # 等效于 101.325 kPa

# 初始猜测值基于你的源数据进行了优化: 
# 富萃取剂相(x0) 主要是环丁砜；富稀释剂相(w0) 主要是正庚烷
for T in temps_sys2:
    # 1. 运行 UNIFAC 模型仿真
    generate_broad_phase_envelope(2, "UNIFAC", mod2_uni, T=T, P=P_sys2, solute_idx=1,
                                  x0_guess=[0.012, 0.010, 0.978], w0_guess=[0.960, 0.033, 0.007])
    
    # 2. 运行 NRTL 模型仿真
    generate_broad_phase_envelope(2, "NRTL", mod2_nrtl, T=T, P=P_sys2, solute_idx=1,
                                  x0_guess=[0.012, 0.010, 0.978], w0_guess=[0.960, 0.033, 0.007])