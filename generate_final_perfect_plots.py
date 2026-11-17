#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт полной генерации канонических, математически точных графиков для Лабораторной работы №5
в строгом соответствии с требованиями преподавателя:
1. Моделирование продлено до момента полной асимптотической синхронизации (log10(e) ~ -12 ... -14, t до 120-340 с).
2. Честные непрерывные временные ряды координат и логарифмическая ошибка log10(||e(t)||).
3. Строгий научный стиль: ТОЛЬКО НАИМЕНОВАНИЯ ОСЕЙ И НАЗВАНИЕ СВЕРХУ (без текстовых выносок, стрелок, плашек).
4. В двунаправленной синхронизации — двумерная карта варьирования параметров (k1, k2) + семейство кривых ошибки.
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

from sprott_sync import sprott_step_cd, sprott_step_coupled, sprott_step_bidir

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

C_MASTER = '#111111'
C_X = '#1f77b4'       # Синий
C_Y = '#d62728'       # Красный
C_Z = '#2ca02c'       # Зеленый
C_PURPLE = '#8e44ad'  # Пурпурный

h = 0.005

# Начальные условия
# Ведущая система на аттракторе
xm0, ym0, zm0 = 0.1, 0.1, 0.1
for _ in range(5000):
    xm0, ym0, zm0 = sprott_step_cd(xm0, ym0, zm0, h)

xs0 = xm0 + 0.5
ys0 = ym0 - 0.4
zs0 = zm0 + 0.3

print("=" * 80)
print("ГЕНЕРАЦИЯ ПОЛНОГО НАБОРА ГРАФИКОВ ПО ТРЕБОВАНИЯМ ПРЕПОДАВАТЕЛЯ")
print("=" * 80)

# =========================================================================
# 1. РИС. 1: Однонаправленная синхронизация - логарифм установившейся ошибки от k
# =========================================================================
print("-> 1. Построение Рис. 1 (Логарифм ошибки от k при однонаправленной связи)...")
k_range = np.linspace(0.0, 4.0, 81)
s_trans = int(120.0 / h)
s_eval = int(20.0 / h)

errs_k_x = np.zeros(len(k_range))
errs_k_y = np.zeros(len(k_range))
errs_k_z = np.zeros(len(k_range))

for ik, kval in enumerate(k_range):
    # Канал X
    xm, ym, zm = xm0, ym0, zm0
    xs, ys, zs = xs0, ys0, zs0
    for _ in range(s_trans):
        xs, ys, zs = sprott_step_coupled(xs, ys, zs, xm, ym, zm, h, kval, 0)
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)
    e_accum = 0.0
    for _ in range(s_eval):
        xs, ys, zs = sprott_step_coupled(xs, ys, zs, xm, ym, zm, h, kval, 0)
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)
        e_accum += np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
    errs_k_x[ik] = e_accum / s_eval

    # Канал Y
    xm, ym, zm = xm0, ym0, zm0
    xs, ys, zs = xs0, ys0, zs0
    for _ in range(s_trans):
        xs, ys, zs = sprott_step_coupled(xs, ys, zs, xm, ym, zm, h, kval, 1)
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)
    e_accum = 0.0
    for _ in range(s_eval):
        xs, ys, zs = sprott_step_coupled(xs, ys, zs, xm, ym, zm, h, kval, 1)
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)
        e_accum += np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
    errs_k_y[ik] = e_accum / s_eval

    # Канал Z
    xm, ym, zm = xm0, ym0, zm0
    xs, ys, zs = xs0, ys0, zs0
    for _ in range(s_trans):
        xs, ys, zs = sprott_step_coupled(xs, ys, zs, xm, ym, zm, h, kval, 2)
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)
    e_accum = 0.0
    for _ in range(s_eval):
        xs, ys, zs = sprott_step_coupled(xs, ys, zs, xm, ym, zm, h, kval, 2)
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)
        e_accum += np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
    errs_k_z[ik] = e_accum / s_eval

log_errs_k_x = np.log10(np.clip(errs_k_x, 1e-16, 1e2))
log_errs_k_y = np.log10(np.clip(errs_k_y, 1e-16, 1e2))
log_errs_k_z = np.log10(np.clip(errs_k_z, 1e-16, 1e2))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(k_range, log_errs_k_x, color=C_X, lw=2.2, label='Канал связи X')
ax.plot(k_range, log_errs_k_y, color=C_Y, lw=2.2, linestyle='--', label='Канал связи Y')
ax.plot(k_range, log_errs_k_z, color=C_Z, lw=2.2, linestyle='-.', label='Канал связи Z')
ax.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
ax.set_title('Однонаправленная синхронизация: зависимость установившейся ошибки от силы связи k', fontweight='bold', pad=10)
ax.set_xlabel('Коэффициент связи k', fontweight='bold')
ax.set_ylabel(r'$\log_{10}(\|e\|)$', fontweight='bold')
ax.set_xlim(0, 4.0)
ax.set_ylim(-16.8, 3.0)
ax.set_yticks(np.arange(-16, 4, 2))
ax.legend(loc='lower left', framealpha=0.90)
plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, "01_unidirectional_error_vs_k.png")
plt.savefig(p1)
plt.close()
print(f"-> Сохранено: {p1}")

# =========================================================================
# 2. РИС. 2, 3, 4: Временные реализации и ошибка ДО ПОЛНОЙ СХОДИМОСТИ (-14)
# =========================================================================
# X: k = 2.0, t = 250 с (полная сходимость к 10^-14 на 92.5 с и стабильное удержание)
# Y: k = 2.0, t = 400 с (длительное моделирование, подтверждающее отсутствие синхронизации)
# Z: k = 0.8, t = 400 с (сходимость к 10^-14 на 286.8 с и стабильное удержание)

configs_unidir = [
    (0, 'X', C_X, 2.0, 250.0, "02_unidirectional_timeseries_x.png"),
    (1, 'Y', C_Y, 2.0, 400.0, "02_unidirectional_timeseries_y.png"),
    (2, 'Z', C_Z, 0.8, 400.0, "02_unidirectional_timeseries_z.png")
]

for var_idx, vname, vcol, kval, t_max, filename in configs_unidir:
    print(f"-> Построение графика временных рядов канала {vname} (t = {t_max} с, k = {kval})...")
    steps = int(t_max / h)
    time_arr = np.linspace(0, t_max, steps + 1)

    m_series = np.zeros(steps + 1)
    s_series = np.zeros(steps + 1)
    log_errs = np.zeros(steps + 1)

    xm, ym, zm = xm0, ym0, zm0
    xs, ys, zs = xs0, ys0, zs0

    coords_m = [xm, ym, zm]
    coords_s = [xs, ys, zs]
    m_series[0] = coords_m[var_idx]
    s_series[0] = coords_s[var_idx]
    log_errs[0] = np.log10(np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2))

    for s in range(1, steps + 1):
        xs, ys, zs = sprott_step_coupled(xs, ys, zs, xm, ym, zm, h, kval, var_idx)
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)
        
        coords_m = [xm, ym, zm]
        coords_s = [xs, ys, zs]
        m_series[s] = coords_m[var_idx]
        s_series[s] = coords_s[var_idx]
        
        e = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
        log_errs[s] = np.log10(max(e, 1e-16))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})

    ax1.plot(time_arr, m_series, color=C_MASTER, lw=1.1, label=f'Ведущая система ${vname.lower()}_m(t)$', alpha=0.85)
    ax1.plot(time_arr, s_series, color=vcol, linestyle='--', lw=1.1, label=f'Ведомая система ${vname.lower()}_s(t)$', alpha=0.85)
    title_word = 'синхронизация' if var_idx in [0, 2] else 'связь'
    ax1.set_title(f'Однонаправленная {title_word} по каналу {vname} ($k = {kval}$)', fontweight='bold', pad=10)
    ax1.set_ylabel(f'Координата ${vname.lower()}(t)$', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95)

    ax2.plot(time_arr, log_errs, color=vcol, lw=1.8, label=r'$\log_{10}(\|e(t)\|)$')
    ax2.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
    ax2.set_xlabel('Время t, с', fontweight='bold')
    ax2.set_ylabel(r'$\log_{10}(\|e(t)\|)$', fontweight='bold')
    ax2.set_ylim(-16.5, 1.5)
    ax2.set_yticks(np.arange(-16, 2, 2))
    ax2.legend(loc='upper right', framealpha=0.95)

    plt.tight_layout()
    p_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(p_path)
    plt.close()
    print(f"-> Сохранено: {p_path}")

# =========================================================================
# 3. РИС. 5: ДВУНАПРАВЛЕННАЯ СИНХРОНИЗАЦИЯ: ВАРЬИРОВАНИЕ k1 и k2
# =========================================================================
print("-> 3. Построение Рис. 5: Двумерная карта (k1, k2) и кривые ошибки...")

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
cbar.set_label(r'$\log_{10}(\|e\|)$', fontsize=11)

ax_map.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.5, alpha=0.7)
ax_map.set_title('Влияние коэффициентов на ошибку', fontweight='bold', pad=10)
ax_map.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
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

# =========================================================================
# 4. РИС. 6, 7, 8: Двунаправленная синхронизация по X, Y, Z (до -14)
# =========================================================================
# X: k = 0.8, t = 250 с (сходится к 10^-14 на 101 с и стабильно удерживается)
# Y: k = 0.8, t = 400 с (длительное моделирование, ошибка ~3)
# Z: k = 0.6, t = 400 с (сходится к 10^-14 на 309 с и стабильно удерживается)

configs_bidir = [
    (0, 'X', C_X, 0.8, 250.0, "07_bidirectional_timeseries_x.png"),
    (1, 'Y', C_Y, 0.8, 400.0, "07_bidirectional_timeseries_y.png"),
    (2, 'Z', C_Z, 0.6, 400.0, "07_bidirectional_timeseries_z.png")
]

for var_idx, vname, vcol, kval, t_max, filename in configs_bidir:
    print(f"-> Построение двунаправленного графика канала {vname} (t = {t_max} с, k={kval})...")
    steps = int(t_max / h)
    time_arr = np.linspace(0, t_max, steps + 1)

    m_series = np.zeros(steps + 1)
    s_series = np.zeros(steps + 1)
    log_errs = np.zeros(steps + 1)

    xm, ym, zm = xm0, ym0, zm0
    xs, ys, zs = xs0, ys0, zs0

    coords_m = [xm, ym, zm]
    coords_s = [xs, ys, zs]
    m_series[0] = coords_m[var_idx]
    s_series[0] = coords_s[var_idx]
    log_errs[0] = np.log10(np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2))

    for s in range(1, steps + 1):
        xm, ym, zm, xs, ys, zs = sprott_step_bidir(xm, ym, zm, xs, ys, zs, h, kval, var_idx)
        
        coords_m = [xm, ym, zm]
        coords_s = [xs, ys, zs]
        m_series[s] = coords_m[var_idx]
        s_series[s] = coords_s[var_idx]
        
        e = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
        log_errs[s] = np.log10(max(e, 1e-16))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})

    ax1.plot(time_arr, m_series, color=C_MASTER, lw=1.1, label=f'Система 1 ${vname.lower()}_1(t)$', alpha=0.85)
    ax1.plot(time_arr, s_series, color=vcol, linestyle='--', lw=1.1, label=f'Система 2 ${vname.lower()}_2(t)$', alpha=0.85)
    title_word = 'синхронизация' if var_idx in [0, 2] else 'связь'
    ax1.set_title(f'Двунаправленная {title_word} по каналу {vname} ($k = {kval}$)', fontweight='bold', pad=10)
    ax1.set_ylabel(f'Координата ${vname.lower()}(t)$', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95)

    ax2.plot(time_arr, log_errs, color=vcol, lw=1.8, label=r'$\log_{10}(\|e(t)\|)$')
    ax2.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
    ax2.set_xlabel('Время t, с', fontweight='bold')
    ax2.set_ylabel(r'$\log_{10}(\|e(t)\|)$', fontweight='bold')
    ax2.set_ylim(-16.5, 1.5)
    ax2.set_yticks(np.arange(-16, 2, 2))
    ax2.legend(loc='upper right', framealpha=0.95)

    plt.tight_layout()
    p_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(p_path)
    plt.close()
    print(f"-> Сохранено: {p_path}")

# =========================================================================
# 5. РИС. 9: ОБРАТИМАЯ СИНХРОНИЗАЦИЯ ПО ИТЕРАЦИЯМ
# =========================================================================
print("-> 5. Построение Рис. 9 (Обратимая синхронизация по итерациям)...")
from generate_four_convergence_plots import log_e_x, log_e_y, log_e_z, log_u_x, log_u_y, log_u_z, log_eps

generations = np.arange(len(log_e_x))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.plot(generations, log_e_x, color=C_X, marker='o', markersize=4.5, lw=2.0, label=r'Канал X (точка $X^*_X$)')
ax1.plot(generations, log_e_y, color=C_Y, marker='s', markersize=4.5, lw=2.0, label=r'Канал Y (точка $X^*_Y$)')
ax1.plot(generations, log_e_z, color=C_Z, marker='^', markersize=4.5, lw=2.0, label=r'Канал Z (точка $X^*_Z$)')
ax1.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
ax1.set_title('А. Специализированные точки $X^*_X, X^*_Y, X^*_Z$', fontweight='bold')
ax1.set_xlabel('Номер итерации i', fontweight='bold')
ax1.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
ax1.set_xlim(-0.5, 30.5)
ax1.set_ylim(-16.8, 0.5)
ax1.set_yticks(np.arange(-16, 2, 2))
ax1.legend(loc='lower left', framealpha=0.90)

ax2.plot(generations, log_u_x, color=C_X, lw=2.2, label='Канал связи X')
ax2.plot(generations, log_u_y, color=C_Y, linestyle='--', lw=2.2, label='Канал связи Y')
ax2.plot(generations, log_u_z, color=C_Z, linestyle='-.', lw=2.2, label='Канал связи Z')
ax2.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
ax2.set_title('Б. Универсальная точка $X^*_{univ}$ ($W_T = 3.0$ с, $K = 2.0$)', fontweight='bold')
ax2.set_xlabel('Номер итерации i', fontweight='bold')
ax2.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
ax2.set_xlim(-0.5, 30.5)
ax2.set_ylim(-16.8, 0.5)
ax2.set_yticks(np.arange(-16, 2, 2))
ax2.legend(loc='lower left', framealpha=0.90)

plt.suptitle('Сходимость обратимой синхронизации по итерациям', fontweight='bold', y=0.99)
plt.tight_layout()
p9 = os.path.join(PLOTS_DIR, "09_reversible_convergence_vs_iterations.png")
plt.savefig(p9)
plt.close()
print(f"-> Сохранено: {p9}")

# =========================================================================
# 6. РИС. 10: Влияние длины окна WT
# =========================================================================
print("-> 6. Построение Рис. 10 (Влияние длины окна WT)...")
from generate_four_convergence_plots import run_exact_simulation

pt_univ = [-1.286205, -1.101576, 0.025948]
k_univ = 2.0
windows = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
styles = ['-', '--', '-.', ':', '-', '--']
colors_wt = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

fig, ax = plt.subplots(figsize=(9, 5.5))
for wt, st, c in zip(windows, styles, colors_wt):
    errs = run_exact_simulation(pt_univ, wt, k_univ, 0)
    log_errs = np.log10(np.clip(errs[:26], 1e-16, 1e2))
    ax.plot(range(26), log_errs, color=c, linestyle=st, lw=2.0, label=f'$W_T = {wt}$ с')

ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
ax.set_title('Зависимость сходимости ошибки от длины временного окна $W_T$ ($K = 2.0$)', fontweight='bold', pad=10)
ax.set_xlabel('Номер итерации i', fontweight='bold')
ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
ax.set_xlim(-0.5, 25.5)
ax.set_ylim(-16.8, 3.0)
ax.set_yticks(np.arange(-16, 4, 2))
ax.legend(loc='lower left', framealpha=0.90)
plt.tight_layout()
p10 = os.path.join(PLOTS_DIR, "10_reversible_window_comparison.png")
plt.savefig(p10)
plt.close()
print(f"-> Сохранено: {p10}")

print("\n" + "=" * 80)
print("ВСЕ ГРАФИКИ УСПЕШНО СФОРМИРОВАНЫ В ПОЛНОМ СООТВЕТСТВИИ С ТРЕБОВАНИЯМИ!")
print("=" * 80)
