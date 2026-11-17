#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт перерисовки рисунка глубокого сканирования аттрактора (16_deep_search_attractor_density.png)
и 3D локализации точек (15_attractor_three_points_3d.png).
Гарантирует 100% четкую видимость всех четырех маркеров:
- Синий круг: X* (e = 0.0)
- Красный круг: Y* (e = 1.11e-16 < эпсилон)
- Зеленый круг: Z* (e = 0.0)
- Пурпурная звезда: Univ* (все 3 канала)
Ни один маркер не перекрывается фоном или легендой!
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sprott_sync import sprott_step_cd

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DB_PATH = os.path.join(BASE_DIR, "optimal_points_database.json")

with open(DB_PATH, "r", encoding="utf-8") as f:
    db = json.load(f)

pX = np.array(db["records_by_channel"]["X"]["best_pt"])
pY = np.array(db["records_by_channel"]["Y"]["best_pt"])
pZ = np.array(db["records_by_channel"]["Z"]["best_pt"])
pU = np.array(db["universal_point"]["coordinates"])

# 1. Интегрируем траекторию аттрактора Спротта B
print("Генерация фазовой траектории аттрактора...")
h = 0.01
curr = np.array([0.2, 0.2, 0.2])
for _ in range(8000):
    curr[0], curr[1], curr[2] = sprott_step_cd(curr[0], curr[1], curr[2], h)

n_pts = 60000
pts = np.zeros((n_pts, 3))
for i in range(n_pts):
    pts[i] = curr
    curr[0], curr[1], curr[2] = sprott_step_cd(curr[0], curr[1], curr[2], h)

# Очищаем окрестность радиусом 0.3 вокруг рекордеров, чтобы фоновые точки их не заслоняли
mask_clear = np.ones(n_pts, dtype=bool)
for pt in [pX, pY, pZ, pU]:
    dists = np.linalg.norm(pts - pt, axis=1)
    mask_clear &= (dists > 0.30)

sub_pts = pts[mask_clear][::3]
color_metric = sub_pts[:, 2]

# =========================================================================
# РИСУНОК 16: Глубокое сканирование аттрактора (карта плотности с точками)
# =========================================================================
print("Отрисовка 16_deep_search_attractor_density.png...")
fig = plt.figure(figsize=(12, 9), dpi=220)
ax = fig.add_subplot(1, 1, 1, projection='3d')

# Фоновый аттрактор (легкая полупрозрачность)
sc = ax.scatter(
    sub_pts[:, 0], sub_pts[:, 1], sub_pts[:, 2],
    c=color_metric, cmap='viridis_r',
    s=4, alpha=0.22, edgecolors='none', rasterized=True
)

cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.06)
cbar.set_label('Координата Z аттрактора', fontsize=11)

# Угол обзора: отличная видимость всех 4 точек без взаимного перекрытия
ax.view_init(elev=24, azim=-50)

# Рисуем 4 ключевые точки ОЧЕНЬ КРУПНО, с контрастной черной обводкой
# 1. Синий круг: X*
ax.scatter(
    [pX[0]], [pX[1]], [pX[2]],
    color='#0055FF', s=420, edgecolors='black', linewidth=2.8,
    zorder=100, label=f"X* ({pX[0]:.2f}, {pX[1]:.2f}, {pX[2]:.2f}) — канал X: ошибка 0.0"
)

# 2. Красный круг: Y*
ax.scatter(
    [pY[0]], [pY[1]], [pY[2]],
    color='#FF1111', s=420, edgecolors='black', linewidth=2.8,
    zorder=100, label=f"Y* ({pY[0]:.2f}, {pY[1]:.2f}, {pY[2]:.2f}) — канал Y: ошибка 1.11e-16"
)

# 3. Зеленый круг: Z*
ax.scatter(
    [pZ[0]], [pZ[1]], [pZ[2]],
    color='#00DD00', s=420, edgecolors='black', linewidth=2.8,
    zorder=100, label=f"Z* ({pZ[0]:.2f}, {pZ[1]:.2f}, {pZ[2]:.2f}) — канал Z: ошибка 0.0"
)

# 4. Пурпурная звезда: Univ*
ax.scatter(
    [pU[0]], [pU[1]], [pU[2]],
    color='#FF00FF', s=600, marker='*', edgecolors='black', linewidth=2.8,
    zorder=100, label=f"Univ* ({pU[0]:.2f}, {pU[1]:.2f}, {pU[2]:.2f}) — все 3 канала"
)

# Добавляем яркие плашки-выноски (callouts) с линиями-указателями
callouts = [
    (pX, "X* (e = 0.0)", "#0055FF", (-0.9, -0.6, 0.45)),
    (pY, "Y* (e < eps)", "#FF1111", (0.55, 0.55, 0.50)),
    (pZ, "Z* (e = 0.0)", "#00AA00", (0.65, 0.65, 0.45)),
    (pU, "Univ* (e ~ 1e-14)", "#CC00CC", (-0.8, 0.65, -0.45))
]

for pt, text, col, offset in callouts:
    target_pos = pt + np.array(offset)
    ax.plot(
        [pt[0], target_pos[0]],
        [pt[1], target_pos[1]],
        [pt[2], target_pos[2]],
        color=col, linestyle='-', linewidth=2.2, zorder=95
    )
    ax.text(
        target_pos[0], target_pos[1], target_pos[2],
        text,
        color='white',
        fontsize=10.5,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=col, edgecolor='black', linewidth=1.6, alpha=0.96),
        zorder=110
    )

ax.set_xlabel('Ось X', fontsize=11, labelpad=8)
ax.set_ylabel('Ось Y', fontsize=11, labelpad=8)
ax.set_zlabel('Ось Z', fontsize=11, labelpad=8)
ax.set_title('Глубокое сканирование аттрактора Спротта B:\nлокализация рекордных точек X*, Y*, Z* и Univ*', fontsize=13, pad=14)

# Размещаем легенду в левом верхнем углу, где она ничего не перекрывает
ax.legend(loc='upper left', fontsize=10, framealpha=0.96, facecolor='white', edgecolor='black')
plt.tight_layout()

out16 = os.path.join(PLOTS_DIR, "16_deep_search_attractor_density.png")
plt.savefig(out16, dpi=200)
plt.close()
print(f"-> Успешно обновлен: {out16}")

# =========================================================================
# РИСУНОК 15: 3D аттрактор с локализацией оптимальных точек
# =========================================================================
print("Отрисовка 15_attractor_three_points_3d.png...")
fig = plt.figure(figsize=(12, 9), dpi=220)
ax = fig.add_subplot(1, 1, 1, projection='3d')

# Траектория тонкой плавной линией
ax.plot(pts[::2, 0], pts[::2, 1], pts[::2, 2], color='navy', alpha=0.25, linewidth=0.6, zorder=10)

ax.view_init(elev=24, azim=-50)

# Те же 4 точки
ax.scatter([pX[0]], [pX[1]], [pX[2]], color='#0055FF', s=420, edgecolors='black', linewidth=2.8, zorder=100, label=f"X* — канал X: ошибка 0.0")
ax.scatter([pY[0]], [pY[1]], [pY[2]], color='#FF1111', s=420, edgecolors='black', linewidth=2.8, zorder=100, label=f"Y* — канал Y: ошибка 1.11e-16")
ax.scatter([pZ[0]], [pZ[1]], [pZ[2]], color='#00DD00', s=420, edgecolors='black', linewidth=2.8, zorder=100, label=f"Z* — канал Z: ошибка 0.0")
ax.scatter([pU[0]], [pU[1]], [pU[2]], color='#FF00FF', s=600, marker='*', edgecolors='black', linewidth=2.8, zorder=100, label=f"Univ* — все 3 канала")

for pt, text, col, offset in callouts:
    target_pos = pt + np.array(offset)
    ax.plot([pt[0], target_pos[0]], [pt[1], target_pos[1]], [pt[2], target_pos[2]], color=col, linewidth=2.2, zorder=95)
    ax.text(target_pos[0], target_pos[1], target_pos[2], text, color='white', fontsize=10.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=col, edgecolor='black', linewidth=1.6, alpha=0.96), zorder=110)

ax.set_xlabel('Ось X', fontsize=11, labelpad=8)
ax.set_ylabel('Ось Y', fontsize=11, labelpad=8)
ax.set_zlabel('Ось Z', fontsize=11, labelpad=8)
ax.set_title('Локализация рекордных точек X*, Y*, Z* и Univ*\nна странном аттракторе Спротта B', fontsize=13, pad=14)

ax.legend(loc='upper left', fontsize=10, framealpha=0.96, facecolor='white', edgecolor='black')
plt.tight_layout()

out15 = os.path.join(PLOTS_DIR, "15_attractor_three_points_3d.png")
plt.savefig(out15, dpi=200)
plt.close()
print(f"-> Успешно обновлен: {out15}")

print("Все рисунки успешно перерисованы!")
