# -*- coding: utf-8 -*-
"""
问题2: FY1 投放 1 枚烟幕弹干扰 M1, 求使遮蔽时间最长的策略。

两阶段方法:
  阶段A(理想烟幕): 研究起爆点 C0 与起爆时刻 te 对遮蔽时长 T(C0,te) 的影响,
                   求 max T(C0*,te*) —— 先不管无人机能不能送到。
  阶段B(可达最优): 直接在控制量 (θ,v,td,τ) 上优化(严格圆柱口径),
                   起爆点由前向运动学自然确定, 速度/投放约束自动满足。

输出: 航向 θ、速度 v、投放点 Pd、起爆点 C0、遮蔽时长 T_shield。
"""
import sys
import numpy as np
from scipy.optimize import differential_evolution

from smoke_model import (G, V_MISSILE, V_SMOKE_SINK, R_SMOKE, T_SMOKE_EFFECT,
                         V_UAV_MIN, V_UAV_MAX, MISSILE0, UAV0,
                         TRUE_TARGET_CENTER, TRUE_TARGET_BOTTOM,
                         R_TRUE, H_TRUE, FAKE_TARGET,
                         unit, missile_pos, uav_pos, smoke_bomb_pos,
                         dist_point_to_segment_vec, sample_cylinder_surface)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

P0 = UAV0['FY1']                 # FY1 初始 (17800,0,1800)
M0 = MISSILE0['M1']              # M1 初始 (20000,0,2000)
T_ARRIVAL = np.linalg.norm(M0) / V_MISSILE   # 导弹到达假目标时刻 ≈ 67 s


# ============================================================
# 阶段 A: 理想烟幕 —— 给定 (C0, te) 算遮蔽时长
# ============================================================
def shielding_time_C0_te(C0, te, m0, model='center', surf=None, dt=0.02):
    """
    起爆点 C0=(x,y,z), 起爆时刻 te。烟幕球在 te 瞬间形成后以 3m/s 下沉,
    有效窗口 [te, te+20]。返回遮蔽时长(秒)。
    model: 'center' 中心近似 | 'cylinder' 严格圆柱(表面采样全遮挡)
    """
    te = float(te)
    C0 = np.asarray(C0, dtype=float)
    t_end = te + T_SMOKE_EFFECT
    ts = np.arange(te, t_end, dt)
    if ts.size == 0:
        return 0.0
    M = np.array([missile_pos(m0, t) for t in ts])           # (N,3)
    C = C0[None, :] + np.array([0.0, 0.0, -V_SMOKE_SINK])[None, :] * (ts - te)[:, None]

    if model == 'center':
        d = dist_point_to_segment_vec(C, M, TRUE_TARGET_CENTER[None, :])
        return float((d <= R_SMOKE).sum() * dt)

    # 严格圆柱: 每个采样点都被挡才算该时刻遮蔽
    blocked = np.ones(len(ts), dtype=bool)
    for P in surf:
        d = dist_point_to_segment_vec(C, M, P[None, :])
        blocked &= (d <= R_SMOKE)
    return float(blocked.sum() * dt)


# ============================================================
# 阶段 B: 可达性 —— (C0, te) 反解控制量
# ============================================================
def C0_te_to_controls(C0, te):
    """
    由起爆点 C0 和起爆时刻 te 反解 (θ, v, τ, td)。
    返回 dict: feasible 及 violation(不可达时的违约量)。
    """
    C0 = np.asarray(C0, dtype=float)
    dx = C0[0] - P0[0]
    dy = C0[1] - P0[1]
    rho = np.hypot(dx, dy)                    # 水平位移
    theta = np.arctan2(dy, dx)

    z0 = C0[2]
    viol = 0.0
    if z0 > P0[2] + 1e-9:                     # 起爆点高于投放高度 → 不可能
        return dict(feasible=False, theta=theta, v=np.nan, tau=np.nan,
                    td=np.nan, violation=1e9, msg='z0 高于投放高度')

    tau = np.sqrt(2.0 * (P0[2] - z0) / G)     # 起爆延迟
    v = rho / te if te > 1e-9 else np.inf     # 速度
    td = te - tau

    feasible = True
    msg = []
    if not (V_UAV_MIN <= v <= V_UAV_MAX):
        feasible = False
        viol += min(abs(v - V_UAV_MIN), abs(v - V_UAV_MAX))
        msg.append(f'v={v:.1f} 越界[{V_UAV_MIN},{V_UAV_MAX}]')
    if tau < 0 or tau > te:
        feasible = False
        viol += abs(tau - te) if tau > te else abs(tau)
        msg.append(f'tau={tau:.2f} 越界[0,{te:.1f}]')

    return dict(feasible=feasible, theta=theta, v=v, tau=tau, td=td,
                violation=viol, msg='; '.join(msg))


def controls_to_C0_te(theta, v, td, tau):
    """前向: (θ,v,td,τ) -> (C0, te)。用于验证反解正确性。"""
    te = td + tau
    Pd = uav_pos(P0, theta, v, td)
    C0 = smoke_bomb_pos(P0, theta, v, td, te)
    return Pd, C0, te


# ============================================================
# 光滑代理目标: 用 sigmoid 软化"d<=10"的硬指示, 使地形光滑可优化
# ============================================================
def soft_T_C0_te(C0, te, m0, w=2.0, dt=0.02):
    """
    软遮蔽时长: ∫ sigmoid((10-d)/w) dt, 平滑近似遮蔽指示函数。
    优化时最大化它(地形光滑), 最终结果再用硬阈值复算。
    """
    te = float(te)
    C0 = np.asarray(C0, dtype=float)
    ts = np.arange(te, te + T_SMOKE_EFFECT, dt)
    if ts.size == 0:
        return 0.0
    M = np.array([missile_pos(m0, t) for t in ts])
    C = C0[None, :] + np.array([0.0, 0.0, -V_SMOKE_SINK])[None, :] * (ts - te)[:, None]
    d = dist_point_to_segment_vec(C, M, TRUE_TARGET_CENTER[None, :])
    sig = 0.5 * (1.0 + np.tanh((R_SMOKE - d) / w))
    return float(sig.sum() * dt)


def _neg_objective(x, model, surf, dt, penalize_reach):
    """负目标(最小化)。用软目标引导; 若要求可达, 对不可达加惩罚。"""
    x0, y0, z0, te = x
    C0 = np.array([x0, y0, z0])
    obj = -soft_T_C0_te(C0, te, M0, dt=dt)
    if penalize_reach:
        ctrl = C0_te_to_controls(C0, te)
        if not ctrl['feasible']:
            obj += 1000.0 + ctrl['violation']   # 不可达 → 重罚
    return obj


def stageA_optimize(model='center', penalize_reach=False, dt=0.02,
                    popsize=18, maxiter=150, seed=1):
    """在 (x0,y0,z0,te) 上做差分进化(软目标引导), 返回硬阈值复算结果。"""
    surf = sample_cylinder_surface(TRUE_TARGET_BOTTOM, R_TRUE, H_TRUE,
                                   n_theta=24, n_z=5)   # 复算时都要用
    bounds = [
        (0.0, 20000.0),    # x0
        (-50.0, 260.0),    # y0
        (0.0, 1800.0),     # z0
        (0.0, 50.0),       # te
    ]
    res = differential_evolution(
        _neg_objective, bounds,
        args=(model, surf, dt, penalize_reach),
        popsize=popsize, maxiter=maxiter, seed=seed,
        tol=1e-5, polish=False, workers=1, updating='deferred'
    )
    x0, y0, z0, te = res.x
    C0 = np.array([x0, y0, z0])
    # 高精度硬阈值复算(中心 + 圆柱两种口径)
    T_center = shielding_time_C0_te(C0, te, M0, 'center', None, dt=1e-3)
    T_cyl = shielding_time_C0_te(C0, te, M0, 'cylinder', surf, dt=1e-3)
    ctrl = C0_te_to_controls(C0, te)
    return dict(C0=C0, te=te, T_center=T_center, T_cyl=T_cyl,
                model=model, ctrl=ctrl, fun=res.fun)


# ============================================================
# 阶段 B: 可达最优 —— 直接在控制量 (θ,v,td,τ) 上优化(单弹)
# ============================================================
def shield_single(theta, v, td, tau, surf, dt=0.02, t_cap=25.0):
    """单枚烟幕弹(严格圆柱)遮蔽时长。起爆点由前向运动学自然确定。"""
    te = td + tau
    ts = np.arange(te, min(te + T_SMOKE_EFFECT, t_cap), dt)
    if ts.size == 0:
        return 0.0
    M = np.array([missile_pos(M0, t) for t in ts])
    det = smoke_bomb_pos(P0, theta, v, td, te)
    C = det[None, :] + np.array([0.0, 0.0, -V_SMOKE_SINK])[None, :] * (ts - te)[:, None]
    blk = np.ones(len(ts), dtype=bool)
    for P in surf:
        blk &= (dist_point_to_segment_vec(C, M, P[None, :]) <= R_SMOKE)
    return float(blk.sum() * dt)


def stageB_optimize(dt=0.02):
    """
    单弹可达最优: 在控制量 (θ,v,td,τ) 上网格搜索(严格圆柱口径)。
    之前"起爆点必须落在视线上"的参数化会漏掉全局最优, 这里放开约束自由寻优。
    """
    surf = sample_cylinder_surface(TRUE_TARGET_BOTTOM, R_TRUE, H_TRUE,
                                   n_theta=16, n_z=4)
    surf_fine = sample_cylinder_surface(TRUE_TARGET_BOTTOM, R_TRUE, H_TRUE,
                                        n_theta=36, n_z=7)
    best = (-1.0, None)
    for theta in np.arange(0.03, 0.17, 0.01):
        for v in np.arange(70.0, 140.1, 5.0):
            for td in np.arange(0.1, 1.6, 0.1):
                for tau in np.arange(0.0, 0.55, 0.05):
                    T = shield_single(theta, v, td, tau, surf, dt=dt)
                    if T > best[0]:
                        best = (T, (theta, v, td, tau))
    theta, v, td, tau = best[1]
    te = td + tau
    C0 = smoke_bomb_pos(P0, theta, v, td, te)
    Pd = uav_pos(P0, theta, v, td)
    T_center = shielding_time_C0_te(C0, te, M0, 'center', None, dt=1e-3)
    T_cyl = shield_single(theta, v, td, tau, surf_fine, dt=1e-3)
    return dict(theta=theta, v=v, td=td, tau=tau, te=te, C0=C0, Pd=Pd,
                T_center=T_center, T_cyl=T_cyl)


def _report_controls(tag, r):
    print(f'--- {tag} ---')
    print(f'  遮蔽时长: 中心近似 = {r["T_center"]:.4f} s | 严格圆柱 = {r["T_cyl"]:.4f} s')
    print(f'  航向 θ = {np.degrees(r["theta"]):.2f}°, 速度 v = {r["v"]:.2f} m/s')
    print(f'  投弹时刻 td = {r["td"]:.3f} s, 起爆延迟 τ = {r["tau"]:.3f} s, 起爆时刻 te = {r["te"]:.3f} s')
    print(f'  投放点 Pd = ({r["Pd"][0]:.2f}, {r["Pd"][1]:.2f}, {r["Pd"][2]:.2f})')
    print(f'  起爆点 C0 = ({r["C0"][0]:.2f}, {r["C0"][1]:.2f}, {r["C0"][2]:.2f})')
    print()


def _report(tag, r):
    print(f'--- {tag} ---')
    print(f'  遮蔽时长: 中心近似 = {r["T_center"]:.4f} s | 严格圆柱 = {r["T_cyl"]:.4f} s')
    print(f'  起爆点 C0 = ({r["C0"][0]:.2f}, {r["C0"][1]:.2f}, {r["C0"][2]:.2f})')
    print(f'  起爆时刻 te = {r["te"]:.3f} s')
    c = r['ctrl']
    print(f'  反解: θ = {np.degrees(c["theta"]):.2f}°, v = {c["v"]:.2f} m/s, '
          f'τ = {c["tau"]:.3f} s, td = {c["td"]:.3f} s')
    if c['feasible']:
        print('  可达性: 可行 ✓')
        Pd, C0v, tev = controls_to_C0_te(c['theta'], c['v'], c['td'], c['tau'])
        print(f'  前向验证: 投放点 Pd = ({Pd[0]:.2f}, {Pd[1]:.2f}, {Pd[2]:.2f})')
        print(f'           起爆点复算 = ({C0v[0]:.2f}, {C0v[1]:.2f}, {C0v[2]:.2f}), te={tev:.3f}')
    else:
        print(f'  可达性: 不可行 ✗  ({c["msg"]})')
    print()


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    # ---------- 阶段A: 理想烟幕(不约束可达) ----------
    print('========== 阶段A: 理想烟幕(中心近似, 不约束可达) ==========')
    rA_center = stageA_optimize(model='center', penalize_reach=False)
    _report('理想(中心近似)', rA_center)

    print('========== 阶段A: 理想烟幕(严格圆柱, 不约束可达) ==========')
    rA_cyl = stageA_optimize(model='cylinder', penalize_reach=False)
    _report('理想(严格圆柱)', rA_cyl)

    # ---------- 阶段B: 直接在控制量上优化(天然可达) ----------
    print('========== 阶段B: 可达最优(直接优化 θ,v,td,τ) ==========')
    rB = stageB_optimize()
    _report_controls('可达最优策略', rB)
