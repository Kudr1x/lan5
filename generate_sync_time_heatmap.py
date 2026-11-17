#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт построения тепловой карты времени установления синхронизации:
t_sync в плоскости (K2, K1) на интервале [-2, 20] x [-2, 20].

Цветовая шкала — градиент раскаленной лавы (inferno):
- Темный пурпур / гранат: быстрая синхронизация (t_sync ~ 5-15 с)
- Огненно-рыжий: умеренная синхронизация (t_sync ~ 25-45 с)
- Ярко-желтый / белый: медленная околопороговая синхронизация (t_sync ~ 60-90 с)
- Темный базальт (#1e272e): область отсутствия синхронизации / расходимости
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit, prange

BASE_DIR = '/home/bobbert/Документы/leti/mnds/lab4/lab5'
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.5,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

h = 0.005
max_time = 100.0
threshold = 0.01  # порог установления синхронизации (||e|| < 0.01)

@njit(fastmath=True)
def numba_step_bidir_custom(xm, ym, zm, xs, ys, zs, h, k1, k2):
    tau = 0.5 * h
    cm_x = k1 * (xs - xm)
    cs_x = k2 * (xm - xs)

    xm1 = xm + tau * (ym * zm + cm_x)
    ym1 = ym + tau * (xm1 - ym)
    zm1 = zm + tau * (1.0 - xm1 * ym1)

    xs1 = xs + tau * (ys * zs + cs_x)
    ys1 = ys + tau * (xs1 - ys)
    zs1 = zs + tau * (1.0 - xs1 * ys1)

    zm2 = zm1 + tau * (1.0 - xm1 * ym1)
    ym2 = (ym1 + tau * xm1) / (1.0 + tau)
    xm2 = xm1 + tau * (ym2 * zm2 + cm_x)

    zs2 = zs1 + tau * (1.0 - xs1 * ys1)
    ys2 = (ys1 + tau * xs1) / (1.0 + tau)
    xs2 = xs1 + tau * (ys2 * zs2 + cs_x)

    return xm2, ym2, zm2, xs2, ys2, zs2

@njit(parallel=True, fastmath=True)
def compute_sync_time_grid(k1_arr, k2_arr, h, max_time, threshold):
    nk1 = len(k1_arr)
    nk2 = len(k2_arr)
    time_map = np.full((nk2, nk1), np.nan)
    max_steps = int(max_time / h)
    
    for i2 in prange(nk2):
        k2 = k2_arr[i2]
        for i1 in range(nk1):
            k1 = k1_arr[i1]
            xm, ym, zm = 0.5, -0.2, 0.4
            xs, ys, zs = 1.0, 0.3, 0.7
            
            consec = 0
            t_hit = -1.0
            
            for s in range(1, max_steps + 1):
                xm, ym, zm, xs, ys, zs = numba_step_bidir_custom(xm, ym, zm, xs, ys, zs, h, k1, k2)
                if abs(xm) > 1e2 or abs(xs) > 1e2 or np.isnan(xm) or np.isnan(xs):
                    break
                e = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
                if e < threshold:
                    consec += 1
                    if consec == 30:  # удержание ошибки ниже порога не менее 0.15 с
                        t_hit = (s - 30) * h
                        break
                else:
                    consec = 0
            
            if t_hit >= 0:
                time_map[i2, i1] = t_hit
            else:
                time_map[i2, i1] = np.nan
                
    return time_map

n_k = 241
k_vals = np.linspace(-2.0, 20.0, n_k)

print(f"Расчет тепловой карты времени синхронизации {n_k}x{n_k} (диапазон [-2, 20] x [-2, 20])...")
time_map = compute_sync_time_grid(k_vals, k_vals, h, max_time, threshold)

# X - K2, Y - K1
K2_grid, K1_grid = np.meshgrid(k_vals, k_vals)
time_map_plot = time_map.T  # транспонирование: строка - K1, столбец - K2

fig, ax = plt.subplots(figsize=(9.5, 8.0))

# Фон области расходимости / несинхронизации — темный базальт
ax.set_facecolor('#1e272e')

# Непрерывный градиент лавы (inferno) без обводки зон
cf = ax.contourf(
    K2_grid, K1_grid, time_map_plot,
    levels=np.linspace(5, 85, 120),
    cmap='inferno',
    extend='both'
)

# Цветовая шкала
cbar = fig.colorbar(cf, ax=ax, pad=0.03, aspect=25)
cbar.set_label(r'Время достижения синхронизации $t_{sync}$, с', fontsize=12, fontweight='bold')
cbar.set_ticks(np.arange(10, 90, 10))

# Опорные линии
ax.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.6)
ax.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.6)
ax.plot([-2, 20], [-2, 20], color='white', linestyle='--', lw=1.3, alpha=0.7)

# Заголовки и оси без лишнего текста
ax.set_title(r'Время достижения синхронизации в плоскости ($K_1, K_2$)', fontweight='bold', pad=12)
ax.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
ax.set_xlim(-2, 20)
ax.set_ylim(-2, 20)

plt.tight_layout()
p_time = os.path.join(PLOTS_DIR, "06_bidirectional_sync_time_heatmap.png")
plt.savefig(p_time, bbox_inches='tight')
plt.close()

print(f"-> Тепловая карта времени синхронизации успешно сохранена: {p_time}")
