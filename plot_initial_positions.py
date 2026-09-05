# -*- coding: utf-8 -*-
"""
2025 数学建模 A 题：无人机与导弹初始位置可视化
坐标系：以假目标为原点，水平面为 xy 平面，z 为高度。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 初始位置数据 ----------
# 导弹 M1 M2 M3
missiles = {
    "M1": (20000, 0, 2000),
    "M2": (19000, 600, 2100),
    "M3": (18000, -600, 1900),
}
# 无人机 FY1 ~ FY5
drones = {
    "FY1": (17800, 0, 1800),
    "FY2": (12000, 1400, 1400),
    "FY3": (6000, -3000, 700),
    "FY4": (11000, 2000, 1800),
    "FY5": (13000, -2000, 1300),
}

# 目标
decoy = (0, 0, 0)          # 假目标（导弹指向的目标）
real_target = (0, 200, 0)  # 真目标下底面圆心

# ---------- 绘图 ----------
fig = plt.figure(figsize=(14, 6.5))
fig.suptitle("2025 数学建模 A 题：无人机 / 导弹初始位置分布", fontsize=15, fontweight="bold")

# ============ 左图：三维视图 ============
ax3 = fig.add_subplot(1, 2, 1, projection="3d")

# 导弹
mx = [missiles[k][0] for k in missiles]
my = [missiles[k][1] for k in missiles]
mz = [missiles[k][2] for k in missiles]
ax3.scatter(mx, my, mz, s=120, c="crimson", marker="^", depthshade=False,
            label="导弹 (M1~M3)", edgecolors="black", linewidths=0.6, zorder=5)
for k, (x, y, z) in missiles.items():
    ax3.text(x, y, z + 260, k, color="crimson", fontsize=10, fontweight="bold",
             ha="center", va="bottom")

# 无人机
dx = [drones[k][0] for k in drones]
dy = [drones[k][1] for k in drones]
dz = [drones[k][2] for k in drones]
ax3.scatter(dx, dy, dz, s=110, c="royalblue", marker="o", depthshade=False,
            label="无人机 (FY1~FY5)", edgecolors="black", linewidths=0.6, zorder=5)
for k, (x, y, z) in drones.items():
    ax3.text(x, y, z - 320, k, color="royalblue", fontsize=10, fontweight="bold",
             ha="center", va="top")

# 导弹指向假目标的飞行方向（虚线）
for k, (x, y, z) in missiles.items():
    ax3.plot([x, decoy[0]], [y, decoy[1]], [z, decoy[2]],
             linestyle="--", color="crimson", alpha=0.35, linewidth=1.2)

# 假目标（原点）
ax3.scatter(*decoy, s=90, c="green", marker="s", depthshade=False,
            label="假目标(原点)", edgecolors="black", linewidths=0.6)
ax3.text(decoy[0], decoy[1], decoy[2] + 300, "假目标", color="green",
         fontsize=10, fontweight="bold", ha="center")

# 真目标（圆柱：半径 7 m，高 10 m，下底面圆心 (0,200,0)）
rt_x, rt_y, rt_z = real_target
theta = np.linspace(0, 2 * np.pi, 40)
# 圆柱侧面
z_bottom, z_top = rt_z, rt_z + 10
cx = rt_x + 7 * np.cos(theta)
cy = rt_y + 7 * np.sin(theta)
for zz in (z_bottom, z_top):
    ax3.plot(cx, cy, np.full_like(theta, zz), color="darkorange", linewidth=1.4)
# 侧面竖直线（抽样若干根）
for i in range(0, 40, 5):
    ax3.plot([cx[i], cx[i]], [cy[i], cy[i]], [z_bottom, z_top],
             color="darkorange", linewidth=1.0, alpha=0.6)
ax3.scatter([rt_x], [rt_y], [z_top], s=60, c="darkorange", marker="*",
            depthshade=False, label="真目标(圆柱 r=7,h=10)", edgecolors="black",
            linewidths=0.5)
ax3.text(rt_x, rt_y + 250, z_top + 180, "真目标", color="darkorange",
         fontsize=10, fontweight="bold", ha="center")

ax3.set_xlabel("x / m", fontsize=11)
ax3.set_ylabel("y / m", fontsize=11)
ax3.set_zlabel("z (高度) / m", fontsize=11)
ax3.set_title("三维视图", fontsize=13, fontweight="bold")
ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax3.view_init(elev=22, azim=-58)

# ============ 右图：俯视图 (x-y 平面) ============
ax2 = fig.add_subplot(1, 2, 2)

ax2.scatter(mx, my, s=150, c="crimson", marker="^", edgecolors="black",
            linewidths=0.6, label="导弹 (M1~M3)", zorder=5)
for k, (x, y, z) in missiles.items():
    ax2.annotate(f"{k}\nz={z}", (x, y), textcoords="offset points",
                 xytext=(8, 10), color="crimson", fontsize=9, fontweight="bold")

ax2.scatter(dx, dy, s=150, c="royalblue", marker="o", edgecolors="black",
            linewidths=0.6, label="无人机 (FY1~FY5)", zorder=5)
for k, (x, y, z) in drones.items():
    ax2.annotate(f"{k}\nz={z}", (x, y), textcoords="offset points",
                 xytext=(8, -14), color="royalblue", fontsize=9, fontweight="bold")

# 导弹 → 假目标 方向线
for k, (x, y, z) in missiles.items():
    ax2.plot([x, decoy[0]], [y, decoy[1]], linestyle="--", color="crimson",
             alpha=0.3, linewidth=1.2)

# 假目标
ax2.scatter(*decoy[:2], s=130, c="green", marker="s", edgecolors="black",
            linewidths=0.6, label="假目标(原点)", zorder=6)
ax2.annotate("假目标", decoy[:2], textcoords="offset points", xytext=(6, 12),
             color="green", fontsize=9, fontweight="bold")

# 真目标（俯视圆，半径 7 m）
ax2.add_patch(Circle(real_target[:2], 7, facecolor="darkorange", alpha=0.5,
                     edgecolor="darkorange", linewidth=1.5, zorder=3,
                     label="真目标(圆柱 r=7)"))
ax2.annotate("真目标", real_target[:2], textcoords="offset points", xytext=(-6, -16),
             color="darkorange", fontsize=9, fontweight="bold")

ax2.set_xlabel("x / m", fontsize=11)
ax2.set_ylabel("y / m", fontsize=11)
ax2.set_title("俯视图 (x-y 平面)", fontsize=13, fontweight="bold")
ax2.set_aspect("equal")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
ax2.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = "初始位置分布图.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("已保存：", out)
plt.show()
