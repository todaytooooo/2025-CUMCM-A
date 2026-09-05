# -*- coding: utf-8 -*-
"""
问题1 可视化: 俯视图 + 侧视图 + 遮蔽判断曲线
让"导弹 / 烟幕云团 / 真目标"的空间位置关系一目了然。
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from smoke_model import (UAV0, MISSILE0, TRUE_TARGET_CENTER, TRUE_TARGET_BOTTOM,
                         R_TRUE, R_SMOKE, V_SMOKE_SINK, T_SMOKE_EFFECT,
                         uav_pos, missile_pos, smoke_center_pos,
                         dist_point_to_segment_vec)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

u0 = UAV0['FY1']
m0 = MISSILE0['M1']
v = 120.0
theta = float(np.arctan2(-u0[1], -u0[0]))   # 朝假目标 = pi
td, te = 1.5, 5.1
T = TRUE_TARGET_CENTER                        # 真目标中心 (0,200,5)

# 几个关键时刻
t_marks = [5.1, 8.0, 9.0, 9.42, 25.1]
shielded_mask = {8.0: False, 9.0: True, 9.42: True, 25.1: False}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

# ---------- 面板 A: 俯视图 (x-y) ----------
ax = axes[0]
ts = np.linspace(0, 26, 400)
M_all = np.array([missile_pos(m0, t) for t in ts])
ax.plot(M_all[:, 0], M_all[:, 1], 'b-', lw=2, label='导弹 M1 轨迹(朝假目标)')
# 烟幕云团(俯视是半径10的圆, 固定在 x=17188)
smoke_xy = (17188.0, 0.0)
ax.add_patch(plt.Circle(smoke_xy, R_SMOKE, color='orange', alpha=0.35,
                        label='烟幕云团(半径10m)'))
ax.plot(*smoke_xy, 'o', color='darkorange', ms=7)
# 真目标(俯视是半径7的圆, 圆心(0,200))
ax.add_patch(plt.Circle((0, 200), R_TRUE, color='green', alpha=0.5,
                        label='真目标(半径7m)'))
ax.plot(0, 0, 'k^', ms=9, label='假目标(原点)')
# 视线: 几个时刻 导弹->真目标中心
for t in t_marks:
    M = missile_pos(m0, t)
    col = 'g' if shielded_mask.get(t, False) else 'r'
    ax.plot([M[0], T[0]], [M[1], T[1]], ls='--', color=col, alpha=0.7, lw=1.2)
    ax.text(M[0], M[1] - 400, f't={t}s', fontsize=8, color=col, ha='center')
ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
ax.set_title('俯视图 (x-y)：烟幕在 y=0，真目标在 y=200')
ax.axis('equal')
ax.set_xlim(15000, 20000)
ax.set_ylim(-1200, 600)
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

# ---------- 面板 B: 侧视图 (x-z) ----------
ax = axes[1]
ax.plot(M_all[:, 0], M_all[:, 2], 'b-', lw=2, label='导弹 M1 轨迹')
# 烟幕中心下沉轨迹
C_all = np.array([smoke_center_pos(u0, theta, v, td, te, t)
                  for t in np.arange(te, te + T_SMOKE_EFFECT, 0.1)])
ax.plot(C_all[:, 0], C_all[:, 2], 'o-', color='darkorange', ms=2.5,
        label='烟幕中心(下沉)')
# 真目标中心
ax.plot(T[0], T[2], 's', color='green', ms=10, label='真目标中心')
ax.plot(0, 0, 'k^', ms=9, label='假目标(原点)')
for t in t_marks:
    M = missile_pos(m0, t)
    C = smoke_center_pos(u0, theta, v, td, te, t) if t >= te else None
    col = 'g' if shielded_mask.get(t, False) else 'r'
    ax.plot([M[0], T[0]], [M[2], T[2]], ls='--', color=col, alpha=0.7, lw=1.2)
    if C is not None:
        ax.plot([C[0]], [C[2]], 'o', color=col, ms=5)
ax.set_xlabel('x / m')
ax.set_ylabel('z / m')
ax.set_title('侧视图 (x-z)：高度关系')
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(15000, 20000)
ax.set_ylim(0, 2500)
ax.grid(alpha=0.3)

# ---------- 面板 C: d(t) 曲线 ----------
ax = axes[2]
ts = np.arange(te, te + T_SMOKE_EFFECT, 1e-3)
C = np.array([smoke_center_pos(u0, theta, v, td, te, t) for t in ts])
M = np.array([missile_pos(m0, t) for t in ts])
d = dist_point_to_segment_vec(C, M, T[None, :])
ax.plot(ts, d, lw=2)
ax.axhline(R_SMOKE, color='r', ls='--', label='遮蔽阈值 R=10 m')
ax.fill_between(ts, 0, R_SMOKE, where=(d <= R_SMOKE),
                color='g', alpha=0.3, label='有效遮蔽区间')
ax.set_xlabel('时间 t / s')
ax.set_ylabel('d(t)：烟幕中心到视线距离 / m')
ax.set_title('遮蔽判断曲线')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('问题1_空间关系.png', dpi=130)
print('已保存 问题1_空间关系.png')
