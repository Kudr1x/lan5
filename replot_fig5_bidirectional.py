#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Перерисовка Рис. 5 (06_bidirectional_error_vs_k.png) по точным требованиям пользователя:
- У графика слева ось Y: «Коэффициент связи K_1»
- У графика слева ось X: «Коэффициент связи K_2»
- Заголовок графика слева: «Влияние коэффициентов на ошибку»
- Подпись colorbar: «log_10(||e||)»
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
def compute_k1_k2_map(k1_arr, k2_arr, h, s_trans, s_eval):
    nk1 = len(k1_arr)
    nk2 = len(k2_arr)
    err_map = np.zeros((nk2, nk1))
    
    for i2 in prange(nk2):
        k2 = k2_arr[i2]
        for i1 in range(nk1):
            k1 = k1_arr[i1]
            xm, ym, zm = 0.5, -0.2, 0.4
            xs, ys, zs = 1.0, 0.3, 0.7
            
            for _ in range(s_trans):
                xm, ym, zm, xs, ys, zs = numba_step_bidir_custom(xm, ym, zm, xs, ys, zs, h, k1, k2, 0)
            
            acc = 0.0
            for _ in range(s_eval):
                xm, ym, zm, xs, ys, zs = numba_step_bidir_custom(xm, ym, zm, xs, ys, zs, h, k1, k2, 0)
                acc += np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
            err_map[i2, i1] = acc / s_eval
            
    return err_map

n_k = 61
k1_vals = np.linspace(0.0, 2.0, n_k)
k2_vals = np.linspace(0.0, 2.0, n_k)

print("Расчет 2D карты взаимной синхронизации (k1, k2)...")
grid_errs = compute_k1_k2_map(k1_vals, k2_vals, h, int(60.0 / h), int(20.0 / h))
log_grid_errs = np.log10(np.clip(grid_errs, 1e-16, 1e2))

# Сетка: X - K2, Y - K1
K2_grid, K1_grid = np.meshgrid(k2_vals, k1_vals)
log_grid_errs_plot = log_grid_errs.T

# Временные кривые для пар (k1, k2)
t_bidir_curves = 250.0
s_bidir_curves = int(t_bidir_curves / h)
t_arr_curves = np.linspace(0, t_bidir_curves, s_bidir_curves + 1)

pairs_to_test = [
    (0.0, 1.5, r'$k_1 = 0.0,\ k_2 = 1.5$', '#7f7f7f', '-'),
    (0.5, 0.5, r'$k_1 = 0.5,\ k_2 = 0.5$', '#ff7f0e', '--'),
    (0.8, 0.8, r'$k_1 = 0.8,\ k_2 = 0.8$', '#1f77b4', '-'),
    (0.3, 1.2, r'$k_1 = 0.3,\ k_2 = 1.2$', '#2ca02c', '-.'),
    (0.15, 0.15, r'$k_1 = 0.15,\ k_2 = 0.15$', '#d62728', ':')
]

fig, (ax_map, ax_lines) = plt.subplots(1, 2, figsize=(16, 6.8), gridspec_kw={'width_ratios': [1.15, 1]})

cf = ax_map.contourf(K2_grid, K1_grid, log_grid_errs_plot, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
cs = ax_map.contour(K2_grid, K1_grid, log_grid_errs_plot, levels=[-14, -10, -5, -1, 0], colors='white', alpha=0.5, linewidths=0.8)
ax_map.clabel(cs, inline=True, fontsize=8, fmt=r'$10^{%d}$')
cbar = fig.colorbar(cf, ax=ax_map, pad=0.03)
# Подпись просто log_10(||e||)
cbar.set_label(r'$\log_{10}(\|e\|)$', fontsize=11)

ax_map.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.5, alpha=0.7)
# Заголовок слева: Влияние коэффициентов на ошибку
ax_map.set_title('Влияние коэффициентов на ошибку', fontweight='bold', pad=10)
# Ось X: Коэффициент связи K_2
ax_map.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
# Ось Y: Коэффициент связи K_1
ax_map.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')

for k1_p, k2_p, lbl, col_p, lstyle in pairs_to_test:
    xm, ym, zm = xm0, ym0, zm0
    xs, ys, zs = xs0, ys0, zs0
    err_t = np.zeros(s_bidir_curves + 1)
    err_t[0] = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
    for s in range(1, s_bidir_curves + 1):
        xm, ym, zm, xs, ys, zs = numba_step_bidir_custom(xm, ym, zm, xs, ys, zs, h, k1_p, k2_p, 0)
        e = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
        err_t[s] = max(e, 1e-16)
    
    ax_lines.plot(t_arr_curves, np.log10(err_t), color=col_p, linestyle=lstyle, lw=1.8, label=lbl)

ax_lines.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
ax_lines.set_title('Сходимость логарифмической ошибки во времени', fontweight='bold', pad=10)
ax_lines.set_xlabel('Время t, с', fontweight='bold')
ax_lines.set_ylabel(r'$\log_{10}(\|e(t)\|)$', fontweight='bold')
ax_lines.set_ylim(-16.8, 3.0)
ax_lines.set_yticks(np.arange(-16, 4, 2))
ax_lines.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, framealpha=0.95, fontsize=9.5)

plt.subplots_adjust(top=0.92, bottom=0.22, left=0.08, right=0.98, wspace=0.25)
p5 = os.path.join(PLOTS_DIR, "06_bidirectional_error_vs_k.png")
plt.savefig(p5, bbox_inches='tight')
plt.close()
print(f"-> Сохранено: {p5}")
