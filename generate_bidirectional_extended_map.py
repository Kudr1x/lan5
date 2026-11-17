#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт генерации чистой расширенной карты взаимной синхронизации:
K_1, K_2 в диапазоне от -2 до +20.

Требования:
- Убраны лишние подписи, бейджи, прямоугольники и точки с левого графика.
- На правом графике в легенде строго только значения коэффициентов.
- Диапазон изменения: от -2 до 20 по обеим осям.
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

from sprott_sync import sprott_step_cd

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
    'grid.alpha': 0.6,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

C_PURPLE = '#8e44ad'
h = 0.005

# Начальные условия
xm0, ym0, zm0 = 0.1, 0.1, 0.1
for _ in range(5000):
    xm0, ym0, zm0 = sprott_step_cd(xm0, ym0, zm0, h)

xs0 = xm0 + 0.5
ys0 = ym0 - 0.4
zs0 = zm0 + 0.3

@njit(fastmath=True)
def numba_step_bidir_custom(xm, ym, zm, xs, ys, zs, h, k1, k2, var_idx):
    tau = 0.5 * h
    cm_x = k1 * (xs - xm) if var_idx == 0 else 0.0
    cm_y = k1 * (ys - ym) if var_idx == 1 else 0.0
    cm_z = k1 * (zs - zm) if var_idx == 2 else 0.0

    cs_x = k2 * (xm - xs) if var_idx == 0 else 0.0
    cs_y = k2 * (ym - ys) if var_idx == 1 else 0.0
    cs_z = k2 * (zm - zs) if var_idx == 2 else 0.0

    xm1 = xm + tau * (ym * zm + cm_x)
    ym1 = ym + tau * (xm1 - ym + cm_y)
    zm1 = zm + tau * (1.0 - xm1 * ym1 + cm_z)

    xs1 = xs + tau * (ys * zs + cs_x)
    ys1 = ys + tau * (xs1 - ys + cs_y)
    zs1 = zs + tau * (1.0 - xs1 * ys1 + cs_z)

    zm2 = zm1 + tau * (1.0 - xm1 * ym1 + cm_z)
    ym2 = (ym1 + tau * (xm1 + cm_y)) / (1.0 + tau)
    xm2 = xm1 + tau * (ym2 * zm2 + cm_x)

    zs2 = zs1 + tau * (1.0 - xs1 * ys1 + cs_z)
    ys2 = (ys1 + tau * (xs1 + cs_y)) / (1.0 + tau)
    xs2 = xs1 + tau * (ys2 * zs2 + cs_x)

    return xm2, ym2, zm2, xs2, ys2, zs2

@njit(parallel=True, fastmath=True)
def compute_extended_map(k1_arr, k2_arr, h, s_trans, s_eval):
    nk1 = len(k1_arr)
    nk2 = len(k2_arr)
    err_map = np.zeros((nk2, nk1))
    
    for i2 in prange(nk2):
        k2 = k2_arr[i2]
        for i1 in range(nk1):
            k1 = k1_arr[i1]
            xm, ym, zm = 0.5, -0.2, 0.4
            xs, ys, zs = 1.0, 0.3, 0.7
            blowup = False
            
            for _ in range(s_trans):
                xm, ym, zm, xs, ys, zs = numba_step_bidir_custom(xm, ym, zm, xs, ys, zs, h, k1, k2, 0)
                if abs(xm) > 1e2 or abs(xs) > 1e2 or np.isnan(xm) or np.isnan(xs):
                    blowup = True
                    break
            
            if blowup:
                err_map[i2, i1] = np.nan
            else:
                acc = 0.0
                for _ in range(s_eval):
                    xm, ym, zm, xs, ys, zs = numba_step_bidir_custom(xm, ym, zm, xs, ys, zs, h, k1, k2, 0)
                    if abs(xm) > 1e2 or abs(xs) > 1e2 or np.isnan(xm) or np.isnan(xs):
                        blowup = True
                        break
                    acc += np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
                if blowup:
                    err_map[i2, i1] = np.nan
                else:
                    err_map[i2, i1] = acc / s_eval
            
    return err_map

n_k = 221  # Сетка 221 x 221 с шагом 0.1 на интервале [-2, 20]
k_vals = np.linspace(-2.0, 20.0, n_k)

print(f"Расчет расширенной карты {n_k}x{n_k} (диапазон [-2, 20] x [-2, 20])...")
grid_errs = compute_extended_map(k_vals, k_vals, h, int(60.0 / h), int(15.0 / h))

# Сетка: X - K2, Y - K1
K2_grid, K1_grid = np.meshgrid(k_vals, k_vals)
log_grid_errs_plot = np.log10(np.clip(grid_errs.T, 1e-16, 1e2))

# Временные кривые для характерных значений
t_curves = 150.0
s_curves = int(t_curves / h)
t_arr_curves = np.linspace(0, t_curves, s_curves + 1)

# В подписях только значения коэффициентов
pairs_to_test = [
    (1.0, 1.0, r'$K_1 = 1.0,\ K_2 = 1.0$', '#1f77b4', '-'),
    (5.0, 5.0, r'$K_1 = 5.0,\ K_2 = 5.0$', '#2ca02c', '-'),
    (15.0, 15.0, r'$K_1 = 15.0,\ K_2 = 15.0$', '#9467bd', '-'),
    (-1.0, 8.0, r'$K_1 = -1.0,\ K_2 = 8.0$', '#ff7f0e', '--'),
    (-1.5, -1.5, r'$K_1 = -1.5,\ K_2 = -1.5$', '#d62728', ':')
]

fig, (ax_map, ax_lines) = plt.subplots(1, 2, figsize=(16.5, 7.2), gridspec_kw={'width_ratios': [1.18, 1]})

# Фон для зоны расходимости (NaN)
ax_map.set_facecolor('#2c3e50')

# Контурная карта
cf = ax_map.contourf(
    K2_grid, K1_grid, log_grid_errs_plot,
    levels=np.linspace(-15, 1, 33),
    cmap='viridis_r',
    extend='both'
)

cs = ax_map.contour(
    K2_grid, K1_grid, log_grid_errs_plot,
    levels=[-14, -10, -5, -2, 0],
    colors='white',
    alpha=0.5,
    linewidths=0.8
)
ax_map.clabel(cs, inline=True, fontsize=8, fmt=r'$10^{%d}$')

# Цветовая шкала
cbar = fig.colorbar(cf, ax=ax_map, pad=0.03)
cbar.set_label(r'$\log_{10}(\|e\|)$', fontsize=11)

# Оси координат K1=0 и K2=0
ax_map.axhline(0, color='white', linestyle=':', lw=1.0, alpha=0.6)
ax_map.axvline(0, color='white', linestyle=':', lw=1.0, alpha=0.6)

# Диагональ K1 = K2
ax_map.plot([-2, 20], [-2, 20], color='white', linestyle='--', lw=1.3, alpha=0.7)

# Заголовок и подписи строго без лишнего текста
ax_map.set_title('Влияние коэффициентов на ошибку', fontweight='bold', pad=10)
ax_map.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax_map.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
ax_map.set_xlim(-2, 20)
ax_map.set_ylim(-2, 20)

# Правый график: динамика ошибки
for k1_p, k2_p, lbl, col_p, lstyle in pairs_to_test:
    xm, ym, zm = xm0, ym0, zm0
    xs, ys, zs = xs0, ys0, zs0
    err_t = np.zeros(s_curves + 1)
    err_t[0] = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
    div_idx = -1
    for s in range(1, s_curves + 1):
        xm, ym, zm, xs, ys, zs = numba_step_bidir_custom(xm, ym, zm, xs, ys, zs, h, k1_p, k2_p, 0)
        if abs(xm) > 1e2 or abs(xs) > 1e2 or np.isnan(xm) or np.isnan(xs):
            div_idx = s
            break
        e = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
        err_t[s] = max(e, 1e-16)
    
    if div_idx != -1:
        valid_t = t_arr_curves[:div_idx]
        valid_e = err_t[:div_idx]
        ax_lines.plot(valid_t, np.log10(valid_e), color=col_p, linestyle=lstyle, lw=1.8, label=lbl)
        ax_lines.plot(valid_t[-1], 2.5, marker='x', markersize=8, color=col_p, markeredgewidth=2.0)
    else:
        ax_lines.plot(t_arr_curves, np.log10(err_t), color=col_p, linestyle=lstyle, lw=1.8, label=lbl)

ax_lines.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
ax_lines.set_title('Сходимость логарифмической ошибки во времени', fontweight='bold', pad=10)
ax_lines.set_xlabel('Время t, с', fontweight='bold')
ax_lines.set_ylabel(r'$\log_{10}(\|e(t)\|)$', fontweight='bold')
ax_lines.set_xlim(0, t_curves)
ax_lines.set_ylim(-16.8, 3.0)
ax_lines.set_yticks(np.arange(-16, 4, 2))
ax_lines.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, framealpha=0.95, fontsize=9.5)

plt.subplots_adjust(top=0.92, bottom=0.22, left=0.07, right=0.98, wspace=0.24)
p_out = os.path.join(PLOTS_DIR, "06_bidirectional_error_vs_k_extended_10x10.png")
plt.savefig(p_out, bbox_inches='tight')
plt.close()
print(f"-> Успешно сохранен чистый расширенный график: {p_out}")
