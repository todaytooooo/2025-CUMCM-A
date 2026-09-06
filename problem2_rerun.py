# -*- coding: utf-8 -*-
"""
问题2 重跑: 单弹最优策略 —— 直接在控制量 (θ,v,td,τ) 上自由优化。
之前阶段B把起爆点强制在视线上(受限子空间), 漏掉了全局最优。
这里用差分进化(DE)在严格圆柱模型上全局搜索, 再用细网格交叉验证。
"""
import sys
import numpy as np
from scipy.optimize import differential_evolution

from smoke_model import (V_SMOKE_SINK, R_SMOKE, T_SMOKE_EFFECT, V_UAV_MIN,
                         V_UAV_MAX, MISSILE0, UAV0, TRUE_TARGET_BOTTOM, R_TRUE,
                         H_TRUE, missile_pos, uav_pos, smoke_bomb_pos,
                         dist_point_to_segment_vec, sample_cylinder_surface)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

P0 = UAV0['FY1']
M0 = MISSILE0['M1']


def shield_single(theta, v, td, tau, surf, dt=0.02, t_cap=25.0):
    """单枚烟幕弹(严格圆柱)遮蔽时长。"""
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


def run():
    surf_coarse = sample_cylinder_surface(TRUE_TARGET_BOTTOM, R_TRUE, H_TRUE,
                                          n_theta=20, n_z=5)
    surf_fine = sample_cylinder_surface(TRUE_TARGET_BOTTOM, R_TRUE, H_TRUE,
                                        n_theta=36, n_z=7)

    def neg(x):
        return -shield_single(x[0], x[1], x[2], x[3], surf_coarse, dt=0.02)

    bounds = [(0.0, 0.4), (V_UAV_MIN, V_UAV_MAX), (0.0, 5.0), (0.0, 5.0)]
    print('DE 优化中 (popsize=20, maxiter=250)...', flush=True)
    res = differential_evolution(neg, bounds, popsize=20, maxiter=250, seed=1,
                                 tol=1e-5, polish=False, updating='deferred')
    th, v, td, tau = res.x
    te = td + tau
    C0 = smoke_bomb_pos(P0, th, v, td, te)
    Pd = uav_pos(P0, th, v, td)
    T_fine = shield_single(th, v, td, tau, surf_fine, dt=1e-3)

    print('=== DE 单弹最优 ===', flush=True)
    print(f'θ = {th:.4f} rad ({np.degrees(th):.2f}°)', flush=True)
    print(f'v = {v:.2f} m/s', flush=True)
    print(f'td = {td:.3f} s, τ = {tau:.3f} s, te = {te:.3f} s', flush=True)
    print(f'投放点 Pd = ({Pd[0]:.2f}, {Pd[1]:.2f}, {Pd[2]:.2f})', flush=True)
    print(f'起爆点 C0 = ({C0[0]:.2f}, {C0[1]:.2f}, {C0[2]:.2f})', flush=True)
    print(f'T(细252点, dt=1e-3) = {T_fine:.3f} s', flush=True)

    # 细网格交叉验证(DE最优点附近小范围)
    print('\n=== 细网格交叉验证(DE最优点邻域) ===', flush=True)
    best = (-1, None)
    for th_ in np.arange(max(0.0, th - 0.06), th + 0.07, 0.01):
        for v_ in np.arange(max(70, v - 25), v + 26, 5):
            for td_ in np.arange(max(0.0, td - 0.4), td + 0.41, 0.05):
                for tau_ in np.arange(max(0.0, tau - 0.5), tau + 0.51, 0.05):
                    T = shield_single(th_, v_, td_, tau_, surf_fine, dt=0.01)
                    if T > best[0]:
                        best = (T, (th_, v_, td_, tau_))
    T, (th_, v_, td_, tau_) = best
    print(f'网格最优 T={T:.3f} s  θ={np.degrees(th_):.2f}° v={v_:.1f} '
          f'td={td_:.2f} τ={tau_:.2f}', flush=True)


if __name__ == '__main__':
    run()
