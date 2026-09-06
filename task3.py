# -*- coding: utf-8 -*-
"""
问题3: FY1 投放 3 枚烟幕弹干扰 M1 —— 三枚烟幕"联合遮蔽" + 粗搜索。

决策变量 x = (θ, v, td1, td2, td3, τ1, τ2, τ3)
  - θ: FY1 航向(水平), v: 速度 ∈ [70,140]
  - tdi: 第 i 枚投放时刻, 满足 td2-td1>=1, td3-td2>=1
  - τi: 第 i 枚投放到起爆延迟, tei = tdi + τi

核心: 联合遮蔽判定 —— 时刻 t, 圆柱表面每个点 P 只要被任意一枚有效
      烟幕球(半径10)挡住视线 M(t)->P, 且所有点都被挡, 才算完全遮蔽。

Step 1: 可靠的单次仿真 shield_time(x)      —— 已向量化(对时间), 快
Step 2: 随机采样粗搜索                     —— 用问题2结论收窄范围
"""
import sys
import numpy as np

from smoke_model import (G, V_MISSILE, V_SMOKE_SINK, R_SMOKE, T_SMOKE_EFFECT,
                         V_UAV_MIN, V_UAV_MAX, MISSILE0, UAV0,
                         TRUE_TARGET_CENTER, TRUE_TARGET_BOTTOM,
                         R_TRUE, H_TRUE, missile_pos, uav_pos, smoke_bomb_pos,
                         smoke_center_pos, dist_point_to_segment_vec,
                         sample_cylinder_surface)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

P0 = UAV0['FY1']                    # FY1 (17800,0,1800)
M0 = MISSILE0['M1']                 # M1  (20000,0,2000)
TAU_MAX = np.sqrt(2 * P0[2] / G)    # 最大落体时间 ≈ 19.17 s
FAR = np.array([1e6, 0.0, 0.0])     # 无效烟幕的占位点(距离恒>R)


# ============================================================
# Step 1: 可靠的单次仿真 (向量化)
# ============================================================
def _smoke_centers_vec(theta, v, tds, taus, ts):
    """返回三个烟幕中心 (3, N, 3) 数组, 无效时刻填 FAR。"""
    dets = []
    for td, tau in zip(tds, taus):
        te = td + tau
        det = smoke_bomb_pos(P0, theta, v, td, te)
        C = det[None, :] + np.array([0.0, 0.0, -V_SMOKE_SINK])[None, :] * (ts - te)[:, None]
        active = (ts >= te) & (ts <= te + T_SMOKE_EFFECT)
        C = np.where(active[:, None], C, FAR)
        dets.append(C)
    return np.array(dets)   # (3, N, 3)


def shield_time(x, surf, dt=0.02, t_cap=None):
    """
    单次仿真: 三枚烟幕联合遮蔽的总时长(秒)。向量化对时间, 快。
    t_cap: 积分时间上限(导弹越过无人机可达区后无需再算)。
    """
    theta, v, td1, td2, td3, tau1, tau2, tau3 = x
    tds = [td1, td2, td3]
    taus = [tau1, tau2, tau3]
    tes = [td + tau for td, tau in zip(tds, taus)]

    t0 = min(tes)
    t1 = max(tes) + T_SMOKE_EFFECT
    if t_cap is not None:
        t1 = min(t1, t_cap)
    ts = np.arange(t0, t1, dt)
    if ts.size == 0:
        return 0.0

    M = np.array([missile_pos(M0, t) for t in ts])   # (N,3)
    Cs = _smoke_centers_vec(theta, v, tds, taus, ts)  # (3,N,3)

    all_blocked = np.ones(len(ts), dtype=bool)
    for P in surf:
        pt_blocked = np.zeros(len(ts), dtype=bool)
        for k in range(3):
            d = dist_point_to_segment_vec(Cs[k], M, P[None, :])
            pt_blocked |= (d <= R_SMOKE)
        all_blocked &= pt_blocked
    return float(all_blocked.sum() * dt)


def shield_segments(x, surf, dt=0.02, t_cap=None):
    """返回遮蔽时段列表 [(start,end),...] 及总时长。"""
    theta, v, td1, td2, td3, tau1, tau2, tau3 = x
    tds = [td1, td2, td3]
    taus = [tau1, tau2, tau3]
    tes = [td + tau for td, tau in zip(tds, taus)]
    t0 = min(tes)
    t1 = max(tes) + T_SMOKE_EFFECT
    if t_cap is not None:
        t1 = min(t1, t_cap)
    ts = np.arange(t0, t1, dt)
    M = np.array([missile_pos(M0, t) for t in ts])
    Cs = _smoke_centers_vec(theta, v, tds, taus, ts)
    all_blocked = np.ones(len(ts), dtype=bool)
    for P in surf:
        pt = np.zeros(len(ts), dtype=bool)
        for k in range(3):
            pt |= (dist_point_to_segment_vec(Cs[k], M, P[None, :]) <= R_SMOKE)
        all_blocked &= pt
    segs = []
    if all_blocked.any():
        idx = np.where(all_blocked)[0]
        s = p = idx[0]
        for i in idx[1:]:
            if i == p + 1:
                p = i
            else:
                segs.append((ts[s], ts[p])); s = p = i
        segs.append((ts[s], ts[p]))
    return segs, float(all_blocked.sum() * dt)


# ============================================================
# Step 2: 随机采样粗搜索 (收窄范围)
# ============================================================
def sample_strategies(n, seed=1, theta_range=(0.0, 0.5), td1_max=3.0,
                      gap_range=(1.0, 2.5), tau_max=2.0):
    """
    在问题2启示的收窄范围内随机采样 n 组可行策略。
    收窄依据: 导弹约 7.4s 越过 FY1 可达区, 三弹起爆须落在 [0,~10s] 并时间重叠
    才能形成"联合遮蔽", 否则后弹成废弹。
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(*theta_range, n)
    v = rng.uniform(V_UAV_MIN, V_UAV_MAX, n)
    td1 = rng.uniform(0.0, td1_max, n)
    td2 = td1 + rng.uniform(*gap_range, n)
    td3 = td2 + rng.uniform(*gap_range, n)
    tau1 = rng.uniform(0.0, tau_max, n)
    tau2 = rng.uniform(0.0, tau_max, n)
    tau3 = rng.uniform(0.0, tau_max, n)
    return np.column_stack([theta, v, td1, td2, td3, tau1, tau2, tau3])


def coarse_search(n=4000, dt=0.05, n_theta=16, n_z=4, t_cap=25.0, seed=1):
    """随机采样 n 组策略, 评估联合遮蔽时长, 返回 (scores, X, surf)。"""
    surf = sample_cylinder_surface(TRUE_TARGET_BOTTOM, R_TRUE, H_TRUE,
                                   n_theta=n_theta, n_z=n_z)
    X = sample_strategies(n, seed=seed)
    scores = np.array([shield_time(x, surf, dt=dt, t_cap=t_cap) for x in X])
    return scores, X, surf


def report_best(scores, X, surf, top_k=5, t_cap=25.0):
    """打印前 top_k 个最优策略 + 遮蔽时间段。"""
    order = np.argsort(-scores)[:top_k]
    print(f'共采样 {len(scores)} 组, 正遮蔽 {int((scores > 0).sum())} 组, '
          f'最优 {scores[order[0]]:.3f} s, 均值 {scores.mean():.3f} s, '
          f'中位 {np.median(scores):.3f} s')
    print()
    for rank, i in enumerate(order, 1):
        theta, v, td1, td2, td3, tau1, tau2, tau3 = X[i]
        tds = [td1, td2, td3]
        taus = [tau1, tau2, tau3]
        tes = [td + tau for td, tau in zip(tds, taus)]
        segs, T = shield_segments(X[i], surf, dt=0.01, t_cap=t_cap)
        print(f'[第{rank}名] 遮蔽时长 {T:.3f} s')
        print(f'  θ={np.degrees(theta):.2f}°, v={v:.2f} m/s')
        print(f'  投放时刻 td = [{td1:.2f}, {td2:.2f}, {td3:.2f}] s')
        print(f'  起爆延迟 τ = [{tau1:.2f}, {tau2:.2f}, {tau3:.2f}] s')
        print(f'  起爆时刻 te = [{tes[0]:.2f}, {tes[1]:.2f}, {tes[2]:.2f}] s')
        for k in range(3):
            Pd = uav_pos(P0, theta, v, tds[k])
            C0 = smoke_bomb_pos(P0, theta, v, tds[k], tes[k])
            print(f'    弹{k+1}: 投放点 ({Pd[0]:.0f},{Pd[1]:.0f},{Pd[2]:.0f}) '
                  f'→ 起爆点 ({C0[0]:.0f},{C0[1]:.0f},{C0[2]:.0f})')
        # 单弹贡献(判断是否真联合遮蔽)
        solo = []
        for k in range(3):
            xk = list(X[i])
            xk[2 + k] = tds[k]          # 保留该弹, 其余推到很远(不参与)
            xk[2 + ((k + 1) % 3)] = 50.0
            xk[2 + ((k + 2) % 3)] = 51.0
            solo.append(shield_time(xk, surf, dt=0.01, t_cap=t_cap))
        print(f'  单弹各自贡献: [{solo[0]:.2f}, {solo[1]:.2f}, {solo[2]:.2f}] s '
              f'(联合 {T:.2f} s)')
        print(f'  遮蔽时间段: ' + ', '.join(f'[{a:.2f},{b:.2f}]' for a, b in segs))
        print()


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    surf = sample_cylinder_surface(TRUE_TARGET_BOTTOM, R_TRUE, H_TRUE,
                                   n_theta=16, n_z=4)

    # ---------- Step 1 自检 ----------
    print('========== Step 1: 单次仿真自检 ==========')
    x_test = [0.2, 80.0, 0.5, 2.0, 3.5, 0.4, 0.4, 0.4]
    T = shield_time(x_test, surf, dt=0.02, t_cap=25.0)
    theta, v, td1, td2, td3, tau1, tau2, tau3 = x_test
    tds = [td1, td2, td3]; taus = [tau1, tau2, tau3]
    tes = [td + tau for td, tau in zip(tds, taus)]
    print(f'测试策略: θ={np.degrees(theta):.1f}° v={v} td={tds} τ={taus}')
    for k in range(3):
        Pd = uav_pos(P0, theta, v, tds[k])
        C0 = smoke_bomb_pos(P0, theta, v, tds[k], tes[k])
        print(f'  弹{k+1}: td={tds[k]:.2f} te={tes[k]:.2f} '
              f'投放点=({Pd[0]:.0f},{Pd[1]:.0f},{Pd[2]:.0f}) '
              f'起爆点=({C0[0]:.0f},{C0[1]:.0f},{C0[2]:.0f})')
    print(f'  → 联合遮蔽时长 T = {T:.3f} s\n')

    # ---------- Step 2: 粗搜索 ----------
    print('========== Step 2: 随机采样粗搜索(收窄范围) ==========')
    scores, X, surf2 = coarse_search(n=4000, dt=0.05, seed=1)
    report_best(scores, X, surf2, top_k=5)
