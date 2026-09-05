# -*- coding: utf-8 -*-
"""
2025 高教社杯数学建模 A 题《烟幕干扰弹的投放策略》核心模型

按解读拆分的四部分组织：
  ① 无人机轨迹 U(t)
  ② 导弹轨迹   M(t)
  ③ 烟幕弹/烟幕云团运动 B(t)、C(t)
  ④ 遮蔽判断  (点到线段距离 ≤ 烟幕半径)

真目标有两种处理：
  - 中心近似: 把圆柱近似为质心 T=(0,200,5)         （问题1 的快速版）
  - 严格圆柱: 对整个圆柱表面采样，要求所有点均被遮蔽  （更严格）

作者: 2025-09
依赖: numpy / scipy / matplotlib
"""

import sys

import numpy as np

# Windows 控制台默认 GBK，强制 stdout 用 UTF-8，避免中文乱码
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ========================= 常量 =========================
G = 9.8                        # 重力加速度 m/s^2
V_MISSILE = 300.0              # 导弹速度 m/s
V_SMOKE_SINK = 3.0             # 烟幕云团起爆后下沉速度 m/s
R_SMOKE = 10.0                 # 烟幕有效遮蔽半径 m
T_SMOKE_EFFECT = 20.0          # 起爆后有效遮蔽时间 s
V_UAV_MIN, V_UAV_MAX = 70.0, 140.0   # 无人机速度范围 m/s
DROP_INTERVAL = 1.0            # 同一无人机相邻两枚投放最小间隔 s

# 假目标 = 坐标系原点
FAKE_TARGET = np.array([0.0, 0.0, 0.0])

# 真目标: 圆柱 半径7 高10 下底面圆心(0,200,0)
R_TRUE = 7.0
H_TRUE = 10.0
TRUE_TARGET_BOTTOM = np.array([0.0, 200.0, 0.0])
TRUE_TARGET_CENTER = np.array([0.0, 200.0, H_TRUE / 2.0])   # 中心近似 (0,200,5)

# 初始位置
MISSILE0 = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0]),
}
UAV0 = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0]),
}


# ========================= 几何工具 =========================
def unit(v):
    """单位向量（零向量原样返回）"""
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def dist_point_to_segment(P, A, B):
    """
    点 P 到线段 AB 的距离（标量版）。
    即解读中两个条件 d(t)<=10 与 0<=(C-M)·(T-M)/||T-M||^2<=1 的合并等价形式。
    """
    AB = B - A
    AP = P - A
    L2 = float(AB @ AB)
    if L2 < 1e-12:
        return float(np.linalg.norm(P - A))
    s = min(max(float(AP @ AB) / L2, 0.0), 1.0)
    return float(np.linalg.norm(P - (A + s * AB)))


def dist_point_to_segment_vec(P, A, B):
    """
    向量版: P、A 为 (N,3)，B 为 (3,) 或 (N,3)。
    返回 (N,) 的逐时刻点到线段距离。
    """
    AB = B - A
    AP = P - A
    L2 = np.sum(AB * AB, axis=-1)
    s = np.clip(np.sum(AP * AB, axis=-1) / L2, 0.0, 1.0)
    proj = A + s[:, None] * AB
    return np.linalg.norm(P - proj, axis=-1)


# ========================= ① 无人机轨迹 =========================
def uav_pos(u0, theta, v, t):
    """
    无人机位置。
    u0    : 初始位置 (3,)
    theta : 航向角(弧度, 水平面内, 从 x 正方向逆时针)
    v     : 飞行速度(等高度匀速直线)
    t     : 时刻(从受领任务起)
    """
    return u0 + v * t * np.array([np.cos(theta), np.sin(theta), 0.0])


# ========================= ② 导弹轨迹 =========================
def missile_pos(m0, t):
    """导弹位置: 始终直指假目标(原点), 匀速 V_MISSILE。"""
    return m0 + V_MISSILE * t * unit(-m0)


# ========================= ③ 烟幕弹 / 烟幕云团 =========================
def smoke_bomb_pos(u0, theta, v, td, t):
    """
    烟幕弹位置(投放后, 平抛)。
    投放点 = 无人机在 td 时刻的位置; 水平初速度继承无人机速度, 竖直自由落体。
    仅对 t >= td 有意义。
    """
    release = uav_pos(u0, theta, v, td)
    v0 = v * np.array([np.cos(theta), np.sin(theta), 0.0])
    tau = t - td
    return release + v0 * tau + 0.5 * np.array([0.0, 0.0, -G]) * tau ** 2


def smoke_center_pos(u0, theta, v, td, te, t):
    """
    烟幕云团中心位置。
    起爆前(te 之前)云团尚未形成; 起爆瞬间在起爆点形成球状云团,
    之后以 V_SMOKE_SINK 匀速下沉。仅对 t >= te 有意义。
    """
    det_pos = smoke_bomb_pos(u0, theta, v, td, te)   # 起爆点
    return det_pos + np.array([0.0, 0.0, -V_SMOKE_SINK * (t - te)])


# ========================= ④ 遮蔽判断 =========================
def is_shielded(C, M, T):
    """烟幕球(中心C, 半径R_SMOKE)是否遮蔽导弹 M 到目标点 T 的视线。"""
    return dist_point_to_segment(C, M, T) <= R_SMOKE


def sample_cylinder_surface(center_bottom, r, h, n_theta=48, n_z=9):
    """
    对圆柱表面采样(用于严格遮蔽判断)。
    返回 (N,3) 采样点。表面点决定视线是否被整体遮挡, 故只需采样表面。
    """
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(0, h, n_z)
    pts = []
    for z in zs:
        for th in thetas:
            pts.append([r * np.cos(th), r * np.sin(th), z])
    pts = np.array(pts) + center_bottom
    return pts


def is_shielded_cylinder(C, M, surf_pts):
    """
    严格版: 烟幕球是否遮蔽整个真目标圆柱。
    要求圆柱表面所有采样点到视线线段距离 <= R_SMOKE (即所有点被遮挡)。
    """
    d = dist_point_to_segment_vec(np.tile(C, (len(surf_pts), 1)),
                                  np.tile(M, (len(surf_pts), 1)),
                                  surf_pts)
    return bool(np.all(d <= R_SMOKE))


# ========================= 通用: 遮蔽时长 =========================
def shielding_time_center(u0, theta, v, td, te, m0, T, t_span, dt=1e-3):
    """
    中心近似下, 单枚烟幕弹的遮蔽时长。
    t_span : (t_start, t_end) 有效窗口, 通常为 (te, te+T_SMOKE_EFFECT)。
    返回总遮蔽时长(秒)。
    """
    t0, t1 = t_span
    ts = np.arange(t0, t1 + dt, dt)
    C = np.array([smoke_center_pos(u0, theta, v, td, te, t) for t in ts])
    M = np.array([missile_pos(m0, t) for t in ts])
    d = dist_point_to_segment_vec(C, M, T[None, :])
    return float((d <= R_SMOKE).sum() * dt)


# ========================= 问题 1 =========================
def problem1(use_center=True, dt=1e-3):
    """
    问题1: FY1 以 120 m/s 朝假目标飞行, 受领1.5s后投放, 3.6s后起爆。
    给出对 M1 的有效遮蔽时长。
    use_center: True 用中心近似; False 用严格圆柱表面采样。
    """
    u0 = UAV0['FY1']
    m0 = MISSILE0['M1']
    v = 120.0
    theta = float(np.arctan2(-u0[1], -u0[0]))   # 朝假目标(原点)的水平航向 = pi
    td = 1.5
    te = td + 3.6                               # 起爆时刻 5.1 s

    t0, t1 = te, te + T_SMOKE_EFFECT            # 有效窗口 [5.1, 25.1]

    # 关键中间量
    Pd = uav_pos(u0, theta, v, td)
    C0 = smoke_bomb_pos(u0, theta, v, td, te)

    if use_center:
        T_shield = shielding_time_center(u0, theta, v, td, te, m0,
                                         TRUE_TARGET_CENTER, (t0, t1), dt)
        return dict(td=td, te=te, Pd=Pd, C0=C0, T_shield=T_shield,
                    target='center', T=TRUE_TARGET_CENTER)

    # 严格圆柱采样
    surf = sample_cylinder_surface(TRUE_TARGET_BOTTOM, R_TRUE, H_TRUE)
    ts = np.arange(t0, t1 + dt, dt)
    cnt = 0
    for t in ts:
        C = smoke_center_pos(u0, theta, v, td, te, t)
        M = missile_pos(m0, t)
        if is_shielded_cylinder(C, M, surf):
            cnt += 1
    T_shield = cnt * dt
    return dict(td=td, te=te, Pd=Pd, C0=C0, T_shield=T_shield,
                target='cylinder', T=TRUE_TARGET_CENTER)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 中文字体，避免图表缺字
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    np.set_printoptions(precision=3, suppress=True)

    # ---------- 问题1 中心近似 ----------
    r1 = problem1(use_center=True)
    print('========== 问题1 (中心近似) ==========')
    print(f"投放时刻 td      = {r1['td']} s")
    print(f"起爆时刻 te      = {r1['te']} s")
    print(f"投放点 Pd        = {r1['Pd']}")
    print(f"起爆点 C0        = {r1['C0']}")
    print(f"有效遮蔽时长     = {r1['T_shield']:.4f} s")

    # ---------- 问题1 严格圆柱 ----------
    r1c = problem1(use_center=False)
    print('\n========== 问题1 (严格圆柱采样) ==========')
    print(f"有效遮蔽时长     = {r1c['T_shield']:.4f} s")

    # ---------- 可视化: 距离随时间变化 ----------
    u0 = UAV0['FY1']
    m0 = MISSILE0['M1']
    v = 120.0
    theta = float(np.arctan2(-u0[1], -u0[0]))
    td, te = 1.5, 5.1
    t0, t1 = te, te + T_SMOKE_EFFECT
    ts = np.arange(t0, t1, 1e-3)
    C = np.array([smoke_center_pos(u0, theta, v, td, te, t) for t in ts])
    M = np.array([missile_pos(m0, t) for t in ts])
    d_center = dist_point_to_segment_vec(C, M, TRUE_TARGET_CENTER[None, :])

    plt.figure(figsize=(8, 4.5))
    plt.plot(ts, d_center, label='烟幕中心到视线距离 d(t)')
    plt.axhline(R_SMOKE, color='r', ls='--', label='遮蔽阈值 R=10 m')
    plt.fill_between(ts, 0, R_SMOKE, where=(d_center <= R_SMOKE),
                     color='g', alpha=0.3, label='有效遮蔽区间')
    plt.xlabel('时间 t / s')
    plt.ylabel('d(t) / m')
    plt.title('问题1: 烟幕中心到"导弹→真目标中心"视线距离')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('问题1_遮蔽距离.png', dpi=120)
    print('\n已保存 问题1_遮蔽距离.png')
