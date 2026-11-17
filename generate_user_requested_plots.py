#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Генерация полного набора графиков по запросу пользователя:
1. Две тепловые карты времени синхронизации (шкала лавы inferno, без обводки зон):
   - Для сигнала X (связь по x)
   - Для сигнала Z (связь по z)
   + Сводный 2-панельный график X и Z
2. Графики для «графика Б» (Универсальная точка X*_univ, WT = 3.0 с, K = 2.0):
   - Канал X
   - Канал Y
   - Канал Z
   + Сводный 3-панельный график
3. Графики зависимости сходимости от длины временного окна WT (0.5, 1.0, 1.5, 2.0, 3.0, 4.0 с):
   - Для канала X
   - Для канала Y
   - Для канала Z
   + Сводный 3-панельный график
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

from sprott_sync import sprott_step_cd, sprott_step_coupled

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.55,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

C_X = '#1f77b4'       # Синий (канал X)
C_Y = '#d62728'       # Красный (канал Y)
C_Z = '#2ca02c'       # Зеленый (канал Z)
C_PURPLE = '#8e44ad'  # Пурпурный (машинный эпсилон)

h = 0.005
eps_mach = 2.220446049250313e-16
log_eps = np.log10(eps_mach)

pt_univ = np.array([-1.2862054512135843, -1.1015760138326, 0.025947709050327224])
k_univ = 2.0
wt_univ = 3.0
windows = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
styles = ['-', '--', '-.', ':', '-', '--']
colors_wt = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

# =========================================================================
# БЛОК 1: ДВЕ ТЕПЛОВЫЕ КАРТЫ ДЛЯ СИГНАЛОВ X И Z
# =========================================================================
print("=" * 80)
print("1. РАСЧЕТ ТЕПЛОВЫХ КАРТ ВРЕМЕНИ СИНХРОНИЗАЦИИ ДЛЯ СИГНАЛОВ X И Z")
print("=" * 80)

@njit(fastmath=True)
def step_bidir_channel(xm, ym, zm, xs, ys, zs, h, k1, k2, var_idx):
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
def compute_sync_time_grid_channel(k1_arr, k2_arr, h, max_time, threshold, var_idx):
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
                xm, ym, zm, xs, ys, zs = step_bidir_channel(xm, ym, zm, xs, ys, zs, h, k1, k2, var_idx)
                if abs(xm) > 1e2 or abs(xs) > 1e2 or np.isnan(xm) or np.isnan(xs):
                    break
                e = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
                if e < threshold:
                    consec += 1
                    if consec == 30:
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
max_t_sim = 100.0
thresh_sync = 0.01

print(f"Расчет карты для сигнала X ({n_k}x{n_k})...")
time_map_x = compute_sync_time_grid_channel(k_vals, k_vals, h, max_t_sim, thresh_sync, 0)

print(f"Расчет карты для сигнала Z ({n_k}x{n_k})...")
time_map_z = compute_sync_time_grid_channel(k_vals, k_vals, h, max_t_sim, thresh_sync, 2)

K2_grid, K1_grid = np.meshgrid(k_vals, k_vals)
tmap_plot_x = time_map_x.T
tmap_plot_z = time_map_z.T

# Функция построения одиночной тепловой карты
def plot_single_heatmap(tmap, title_text, filename):
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.set_facecolor('#1e272e')
    cf = ax.contourf(K2_grid, K1_grid, tmap, levels=np.linspace(5, 85, 120), cmap='inferno', extend='both')
    cbar = fig.colorbar(cf, ax=ax, pad=0.03, aspect=25)
    cbar.set_label(r'Время достижения синхронизации $t_{sync}$, с', fontsize=12, fontweight='bold')
    cbar.set_ticks(np.arange(10, 90, 10))

    ax.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.6)
    ax.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.6)
    ax.plot([-2, 20], [-2, 20], color='white', linestyle='--', lw=1.3, alpha=0.7)

    ax.set_title(title_text, fontweight='bold', pad=12)
    ax.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
    ax.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
    ax.set_xlim(-2, 20)
    ax.set_ylim(-2, 20)

    plt.tight_layout()
    out_p = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    print(f"-> Сохранено: {out_p}")

plot_single_heatmap(tmap_plot_x, 'Время достижения синхронизации (сигнал X)', '06_bidirectional_sync_time_heatmap_x.png')
plot_single_heatmap(tmap_plot_z, 'Время достижения синхронизации (сигнал Z)', '06_bidirectional_sync_time_heatmap_z.png')

# Сводный 2-панельный график для X и Z
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 7.5))
for ax, tmap, title in zip([ax1, ax2], [tmap_plot_x, tmap_plot_z], ['А. Сигнал X', 'Б. Сигнал Z']):
    ax.set_facecolor('#1e272e')
    cf = ax.contourf(K2_grid, K1_grid, tmap, levels=np.linspace(5, 85, 120), cmap='inferno', extend='both')
    ax.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.6)
    ax.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.6)
    ax.plot([-2, 20], [-2, 20], color='white', linestyle='--', lw=1.3, alpha=0.7)
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
    ax.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
    ax.set_xlim(-2, 20)
    ax.set_ylim(-2, 20)
    cbar = fig.colorbar(cf, ax=ax, pad=0.03, aspect=25)
    cbar.set_label(r'$t_{sync}$, с', fontsize=11, fontweight='bold')

plt.suptitle('Время достижения синхронизации в плоскости ($K_1, K_2$)', fontweight='bold', y=0.98, fontsize=14)
plt.tight_layout()
out_xz = os.path.join(PLOTS_DIR, '06_bidirectional_sync_time_heatmap_xz_combined.png')
plt.savefig(out_xz, bbox_inches='tight')
plt.close()
print(f"-> Сохранено: {out_xz}")

# =========================================================================
# БЛОК 2: ГРАФИК «Б» (УНИВЕРСАЛЬНАЯ ТОЧКА X*_univ) ДЛЯ X, Y, Z
# =========================================================================
print("\n" + "=" * 80)
print("2. РАСЧЕТ И ПОСТРОЕНИЕ ГРАФИКА «Б» (УНИВЕРСАЛЬНАЯ ТОЧКА) ДЛЯ КАНАЛОВ X, Y, Z")
print("=" * 80)

num_iters_u = 30
generations_u = np.arange(num_iters_u + 1)

def run_reversible_simulation(pt, wt, k, var_idx, n_it=30):
    wt_steps = int(round(wt / h))
    window = np.zeros((wt_steps + 1, 3))
    cx, cy, cz = pt[0], pt[1], pt[2]
    for step in range(wt_steps + 1):
        window[step] = [cx, cy, cz]
        cx, cy, cz = sprott_step_cd(cx, cy, cz, h)

    xs = window[0, 0] + 0.15
    ys = window[0, 1] - 0.15
    zs = window[0, 2] + 0.10

    errs = np.zeros(n_it + 1)
    errs[0] = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)

    for it in range(1, n_it + 1):
        for step in range(wt_steps):
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], h, k, var_idx)
        for step in range(wt_steps, 0, -1):
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], -h, -k, var_idx)
        errs[it] = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)
    return errs

errs_u_x = run_reversible_simulation(pt_univ, wt_univ, k_univ, 0, num_iters_u)
errs_u_y = run_reversible_simulation(pt_univ, wt_univ, k_univ, 1, num_iters_u)
errs_u_z = run_reversible_simulation(pt_univ, wt_univ, k_univ, 2, num_iters_u)

log_u_x = np.log10(np.clip(errs_u_x, 1e-16, 1e2))
log_u_y = np.log10(np.clip(errs_u_y, 1e-16, 1e2))
log_u_z = np.log10(np.clip(errs_u_z, 1e-16, 1e2))

# Одиночные графики для каждого сигнала
channels_data = [
    ('X', log_u_x, C_X, 'o', '09_universal_point_convergence_x.png'),
    ('Y', log_u_y, C_Y, 's', '09_universal_point_convergence_y.png'),
    ('Z', log_u_z, C_Z, '^', '09_universal_point_convergence_z.png')
]

for ch_name, log_err_ch, ch_col, ch_marker, fname in channels_data:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(generations_u, log_err_ch, color=ch_col, marker=ch_marker, markersize=5, lw=2.2, label=f'Канал {ch_name}')
    ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
    ax.set_title(f'Универсальная точка: Канал связи {ch_name} ($W_T = 3.0$ с, $K = 2.0$)', fontweight='bold', pad=10)
    ax.set_xlabel('Номер итерации i', fontweight='bold')
    ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
    ax.set_xlim(-0.5, 30.5)
    ax.set_ylim(-16.8, 0.5)
    ax.set_yticks(np.arange(-16, 2, 2))
    ax.legend(loc='lower left', framealpha=0.90)
    plt.tight_layout()
    out_f = os.path.join(PLOTS_DIR, fname)
    plt.savefig(out_f, bbox_inches='tight')
    plt.close()
    print(f"-> Сохранено: {out_f}")

# Сводный 3-панельный график для универсальной точки
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, ch_name, log_e, col, mk in zip([ax1, ax2, ax3], ['X', 'Y', 'Z'], [log_u_x, log_u_y, log_u_z], [C_X, C_Y, C_Z], ['o', 's', '^']):
    ax.plot(generations_u, log_e, color=col, marker=mk, markersize=5, lw=2.2, label=f'Канал {ch_name}')
    ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'$\varepsilon_{mach}$')
    ax.set_title(f'Канал {ch_name}', fontweight='bold', pad=10)
    ax.set_xlabel('Номер итерации i', fontweight='bold')
    ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
    ax.set_xlim(-0.5, 30.5)
    ax.set_ylim(-16.8, 0.5)
    ax.set_yticks(np.arange(-16, 2, 2))
    ax.legend(loc='lower left', framealpha=0.90)

plt.suptitle('Сходимость ошибки в универсальной точке $X^*_{univ}$ ($W_T = 3.0$ с, $K = 2.0$)', fontweight='bold', y=0.99, fontsize=13)
plt.tight_layout()
out_u_all = os.path.join(PLOTS_DIR, '09_universal_point_convergence_all_3.png')
plt.savefig(out_u_all, bbox_inches='tight')
plt.close()
print(f"-> Сохранено: {out_u_all}")

# =========================================================================
# БЛОК 3: ВЛИЯНИЕ ДЛИНЫ ОКНА WT ДЛЯ КАНАЛОВ X, Y, Z
# =========================================================================
print("\n" + "=" * 80)
print("3. РАСЧЕТ И ПОСТРОЕНИЕ ВЛИЯНИЯ ДЛИНЫ ОКНА WT ДЛЯ КАНАЛОВ X, Y, Z")
print("=" * 80)

num_iters_wt = 26
generations_wt = np.arange(num_iters_wt)

def plot_window_comp_single(var_idx, ch_name, fname):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for wt, st, c in zip(windows, styles, colors_wt):
        errs = run_reversible_simulation(pt_univ, wt, k_univ, var_idx, num_iters_wt)
        log_e = np.log10(np.clip(errs[:num_iters_wt], 1e-16, 1e2))
        ax.plot(generations_wt, log_e, color=c, linestyle=st, lw=2.0, label=f'$W_T = {wt}$ с')

    ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
    ax.set_title(f'Канал связи {ch_name}: Зависимость ошибки от окна $W_T$ ($K = 2.0$)', fontweight='bold', pad=10)
    ax.set_xlabel('Номер итерации i', fontweight='bold')
    ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
    ax.set_xlim(-0.5, 25.5)
    ax.set_ylim(-16.8, 3.0)
    ax.set_yticks(np.arange(-16, 4, 2))
    ax.legend(loc='upper right', framealpha=0.90, ncol=2)
    plt.tight_layout()
    out_w = os.path.join(PLOTS_DIR, fname)
    plt.savefig(out_w, bbox_inches='tight')
    plt.close()
    print(f"-> Сохранено: {out_w}")

plot_window_comp_single(0, 'X', '10_window_comparison_x.png')
plot_window_comp_single(1, 'Y', '10_window_comparison_y.png')
plot_window_comp_single(2, 'Z', '10_window_comparison_z.png')

# Сводный 3-панельный график для окон WT
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.8))
for ax, var_idx, ch_name in zip([ax1, ax2, ax3], [0, 1, 2], ['X', 'Y', 'Z']):
    for wt, st, c in zip(windows, styles, colors_wt):
        errs = run_reversible_simulation(pt_univ, wt, k_univ, var_idx, num_iters_wt)
        log_e = np.log10(np.clip(errs[:num_iters_wt], 1e-16, 1e2))
        ax.plot(generations_wt, log_e, color=c, linestyle=st, lw=1.9, label=f'$W_T = {wt}$ с')

    ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'$\varepsilon_{mach}$')
    ax.set_title(f'Канал связи {ch_name}', fontweight='bold', pad=10)
    ax.set_xlabel('Номер итерации i', fontweight='bold')
    ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
    ax.set_xlim(-0.5, 25.5)
    ax.set_ylim(-16.8, 3.0)
    ax.set_yticks(np.arange(-16, 4, 2))
    ax.legend(loc='upper right', framealpha=0.90, ncol=2, fontsize=8.5)

plt.suptitle('Влияние длины временного окна $W_T$ на сходимость ошибки ($K = 2.0$)', fontweight='bold', y=0.99, fontsize=13)
plt.tight_layout()
out_w_all = os.path.join(PLOTS_DIR, '10_window_comparison_all_3.png')
plt.savefig(out_w_all, bbox_inches='tight')
plt.close()
print(f"-> Сохранено: {out_w_all}")

print("\n" + "=" * 80)
print("ВСЕ ГРАФИКИ УСПЕШНО СГЕНЕРИРОВАНЫ!")
print("=" * 80)
