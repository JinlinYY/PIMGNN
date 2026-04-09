import pandas as pd
import numpy as np
from phasepy import component, mixture
from phasepy import virialgamma 
from phasepy.equilibrium import lle

# ==========================================
# 1. 定义纯组分
# ==========================================
water = component(name='Water', Tc=647.13, Pc=220.55, Zc=0.229, Vc=55.948, w=0.344861, GC={'H2O':1})
dem = component(name='Diethoxymethane', Tc=550.0, Pc=30.0, Zc=0.25, Vc=300.0, w=0.35, GC={'CH3':2, 'CH2':1, 'CH2O':2})
p_xylene = component(name='p-Xylene', Tc=616.2, Pc=35.1, Zc=0.25, Vc=379.0, w=0.32, GC={'ACH':4, 'ACCH3':2})

octane = component(name='Octane', Tc=568.7, Pc=24.9, Zc=0.25, Vc=492.0, w=0.39, GC={'CH3':2, 'CH2':6})
thiophene = component(name='Thiophene', Tc=579.4, Pc=56.7, Zc=0.25, Vc=219.0, w=0.19, GC={'THIOPHE':1})
sulfolane = component(name='Sulfolane', Tc=800.0, Pc=45.0, Zc=0.25, Vc=285.0, w=0.6, GC={'CH2':4, 'TMSO2':1})

n_hexane = component(name='n-Hexane', Tc=507.6, Pc=30.2, Zc=0.25, Vc=370.0, w=0.30, GC={'CH3':2, 'CH2':4})
ethyl_acetate = component(name='Ethyl_acetate', Tc=523.3, Pc=38.8, Zc=0.25, Vc=286.0, w=0.36, GC={'CH3':1, 'CH2':1, 'CH3COO':1})

# ==========================================
# 2. 构造“借壳上市” UNIFAC 注入器
# ==========================================
def build_custom_unifac_model(mix):
    """利用合法体系骗过初始化检查，然后将内部矩阵替换为自定义矩阵"""
    dummy_mix = water + dem + p_xylene
    dummy_mix.unifac()
    model = virialgamma(dummy_mix, virialmodel='ideal_gas', actmodel='unifac')
    
    Rk_db = {'CH3': 0.9011, 'CH2': 0.6744, 'CH3COO': 1.9031, 'TMSO2': 1.6, 'THIOPHE': 2.8}
    Qk_db = {'CH3': 0.848,  'CH2': 0.540,  'CH3COO': 1.728,  'TMSO2': 1.4, 'THIOPHE': 2.4}
    mg_map = {'CH3': 1, 'CH2': 1, 'CH3COO': 11, 'TMSO2': 40, 'THIOPHE': 43}
    a0_db = {
        1:  {1: 0.0,   11: 232.1, 40: 1250.0, 43: 45.0},
        11: {1: 114.8, 11: 0.0,   40: 80.0,   43: 20.0},
        40: {1: 300.0, 11: 50.0,  40: 0.0,    43: -20.0},
        43: {1: -11.0, 11: 15.0,  40: 10.0,   43: 0.0}
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
# 4. 配置模型并运行多温度批量仿真
# ==========================================
a_default = np.array([[0., 0.2, 0.2], [0.2, 0., 0.2], [0.2, 0.2, 0.]])

g_sys1 = np.array([[0., 200., 1500.], [50., 0., 100.], [1200., 30., 0.]]) 
g_sys2 = np.array([[0., 100., 1200.], [-20., 0., 50.], [1000., 30., 0.]]) 
g_sys3 = np.array([[0., 150., 1300.], [50., 0., 100.], [1100., 80., 0.]]) 

# ----------------- 体系 1: 水(0) + DEM(1) + 对二甲苯(2) -----------------
mix1 = water + dem + p_xylene
mix1.unifac()
mod1_uni = virialgamma(mix1, virialmodel='ideal_gas', actmodel='unifac')
mix1.NRTL(a_default, g_sys1, np.zeros_like(g_sys1))
mod1_nrtl = virialgamma(mix1, virialmodel='ideal_gas', actmodel='nrtl')

# 体系 1 温度列表
temps_sys1 = [303.15]
for T in temps_sys1:
    generate_broad_phase_envelope(1, "UNIFAC", mod1_uni, T=T, P=1.013, solute_idx=1,
                                  x0_guess=[0.99, 0.005, 0.005], w0_guess=[0.005, 0.005, 0.99])
    generate_broad_phase_envelope(1, "NRTL", mod1_nrtl, T=T, P=1.013, solute_idx=1,
                                  x0_guess=[0.99, 0.005, 0.005], w0_guess=[0.005, 0.005, 0.99])

# ----------------- 体系 2: 辛烷(0) + 噻吩(1) + 环丁砜(2) -----------------
mix2 = octane + thiophene + sulfolane
mod2_uni = build_custom_unifac_model(mix2)
mix2.NRTL(a_default, g_sys2, np.zeros_like(g_sys2))
mod2_nrtl = virialgamma(mix2, virialmodel='ideal_gas', actmodel='nrtl')

# 体系 2 温度列表
temps_sys2 = [313.15, 323.15, 333.15]
for T in temps_sys2:
    generate_broad_phase_envelope(2, "UNIFAC", mod2_uni, T=T, P=1.013, solute_idx=1,
                                  x0_guess=[0.05, 0.05, 0.90], w0_guess=[0.90, 0.05, 0.05])
    generate_broad_phase_envelope(2, "NRTL", mod2_nrtl, T=T, P=1.013, solute_idx=1,
                                  x0_guess=[0.05, 0.05, 0.90], w0_guess=[0.90, 0.05, 0.05])

# ----------------- 体系 3: 正己烷(0) + 乙酸乙酯(1) + 环丁砜(2) -----------------
mix3 = n_hexane + ethyl_acetate + sulfolane
mod3_uni = build_custom_unifac_model(mix3)
mix3.NRTL(a_default, g_sys3, np.zeros_like(g_sys3))
mod3_nrtl = virialgamma(mix3, virialmodel='ideal_gas', actmodel='nrtl')

# 体系 3 温度列表
temps_sys3 = [303.15, 313.15, 323.15]
for T in temps_sys3:
    generate_broad_phase_envelope(3, "UNIFAC", mod3_uni, T=T, P=1.013, solute_idx=1,
                                  x0_guess=[0.05, 0.05, 0.90], w0_guess=[0.90, 0.05, 0.05])
    generate_broad_phase_envelope(3, "NRTL", mod3_nrtl, T=T, P=1.013, solute_idx=1,
                                  x0_guess=[0.05, 0.05, 0.90], w0_guess=[0.90, 0.05, 0.05])