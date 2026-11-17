#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт полной генерации всех требуемых графиков:
1. Двунаправленная синхронизация для КАЖДОГО сигнала (X, Y, Z):
   - Тепловая карта установившейся ошибки log10(||e||)
   - Тепловая карта времени выхода на синхронизацию t_sync (шкала лавы inferno)
   - Двухпанельный график [Карта ошибки + Временные ряды ошибки log10(||e(t)||)]
2. Исследование влияния длины временного окна WT для КАЖДОГО сигнала (X, Y, Z)
   на 1 000 000 (1M) итерациях с логарифмической шкалой абсцисс.
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
    'grid.alpha': 0.55,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

C_PURPLE = '#8e44ad'
h = 0.005
eps_mach = 2.220446049250313e-16
log_eps = np.log10(eps_mach)

# Начальные условия
xm0, ym0, zm0 = 0.1, 0.1, 0.1
for _ in range(5000):
    xm0, ym0, zm0 = sprott_step_cd(xm0, ym0, zm0, h)

xs0 = xm0 + 0.5
ys0 = ym0 - 0.4
zs0 = zm0 + 0.3

# =========================================================================
# БЛОК 1: ДВУНАПРАВЛЕННАЯ СИНХРОНИЗАЦИЯ (X, Y, Z)
# =========================================================================
print("=" * 80)
print("1. РАСЧЕТ КАРТ ОШИБКИ И ВРЕМЕНИ ДЛЯ СИГНАЛОВ X, Y, Z")
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
def compute_bidir_maps(k1_arr, k2_arr, h, s_trans, s_eval, thresh, var_idx):
    nk1 = len(k1_arr)
    nk2 = len(k2_arr)
    err_map = np.zeros((nk2, nk1))
    time_map = np.full((nk2, nk1), np.nan)
    max_steps = s_trans + s_eval
    
    for i2 in prange(nk2):
        k2 = k2_arr[i2]
        for i1 in range(nk1):
            k1 = k1_arr[i1]
            xm, ym, zm = 0.5, -0.2, 0.4
            xs, ys, zs = 1.0, 0.3, 0.7
            blowup = False
            consec = 0
            t_hit = -1.0
            
            for s in range(1, s_trans + 1):
                xm, ym, zm, xs, ys, zs = step_bidir_channel(xm, ym, zm, xs, ys, zs, h, k1, k2, var_idx)
                if abs(xm) > 1e2 or abs(xs) > 1e2 or np.isnan(xm) or np.isnan(xs):
                    blowup = True
                    break
                e = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
                if e < thresh:
                    consec += 1
                    if consec == 30 and t_hit < 0:
                        t_hit = (s - 30) * h
                else:
                    consec = 0
            
            if blowup:
                err_map[i2, i1] = np.nan
            else:
                acc = 0.0
                for s in range(s_trans + 1, max_steps + 1):
                    xm, ym, zm, xs, ys, zs = step_bidir_channel(xm, ym, zm, xs, ys, zs, h, k1, k2, var_idx)
                    if abs(xm) > 1e2 or abs(xs) > 1e2 or np.isnan(xm) or np.isnan(xs):
                        blowup = True
                        break
                    e = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
                    acc += e
                    if e < thresh:
                        consec += 1
                        if consec == 30 and t_hit < 0:
                            t_hit = (s - 30) * h
                    else:
                        consec = 0
                
                if blowup:
                    err_map[i2, i1] = np.nan
                else:
                    err_map[i2, i1] = acc / s_eval
                    
            if t_hit >= 0:
                time_map[i2, i1] = t_hit
                
    return err_map, time_map

n_k = 101
k_vals = np.linspace(0.0, 2.0, n_k)
K2_grid, K1_grid = np.meshgrid(k_vals, k_vals)

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

channels_bidir = [
    (0, 'X', 'x'),
    (1, 'Y', 'y'),
    (2, 'Z', 'z')
]

for var_idx, vname, vchar in channels_bidir:
    print(f"-> Расчет двунаправленной синхронизации для сигнала {vname}...")
    s_trans = int(120.0 / h) if var_idx == 2 else int(80.0 / h)
    s_eval = int(20.0 / h)
    thresh = 0.05
    
    err_m, time_m = compute_bidir_maps(k_vals, k_vals, h, s_trans, s_eval, thresh, var_idx)
    log_err_plot = np.log10(np.clip(err_m.T, 1e-16, 1e2))
    time_plot = time_m.T

    # 1. Отдельная тепловая карта времени (шкала лавы)
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.set_facecolor('#1e272e')
    t_max_val = np.nanmax(time_plot) if not np.all(np.isnan(time_plot)) else 80.0
    if not np.isnan(t_max_val) and t_max_val > 5.0:
        cf_t = ax.contourf(K2_grid, K1_grid, time_plot, levels=np.linspace(5, min(t_max_val, 120.0), 120), cmap='inferno', extend='both')
        cbar_t = fig.colorbar(cf_t, ax=ax, pad=0.03, aspect=25)
        cbar_t.set_label(r'Время синхронизации $t_{sync}$, с', fontsize=12, fontweight='bold')
    else:
        # Для Y, где порог не достигается
        ax.text(1.0, 1.0, 'Синхронизация не достигается\n(поперечный показатель Ляпунова $\\lambda_\\perp \\approx 0$)',
                color='#f39c12', fontsize=12, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#2c3e50', edgecolor='#f39c12'))
    
    ax.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
    ax.set_title(f'Время достижения синхронизации (сигнал {vname})', fontweight='bold', pad=12)
    ax.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
    ax.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    plt.tight_layout()
    p_time_out = os.path.join(PLOTS_DIR, f"06_bidirectional_sync_time_heatmap_{vchar}.png")
    plt.savefig(p_time_out, bbox_inches='tight')
    plt.close()
    print(f"   Карта времени сохранена: {p_time_out}")

    # 2. Отдельная тепловая карта ошибки
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.set_facecolor('#2c3e50')
    cf_e = ax.contourf(K2_grid, K1_grid, log_err_plot, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
    cs_e = ax.contour(K2_grid, K1_grid, log_err_plot, levels=[-14, -10, -5, -2, 0], colors='white', alpha=0.5, linewidths=0.8)
    ax.clabel(cs_e, inline=True, fontsize=8, fmt=r'$10^{%d}$')
    cbar_e = fig.colorbar(cf_e, ax=ax, pad=0.03, aspect=25)
    cbar_e.set_label(r'$\log_{10}(\|e\|)$', fontsize=12, fontweight='bold')
    ax.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
    ax.set_title(f'Влияние коэффициентов на ошибку (сигнал {vname})', fontweight='bold', pad=12)
    ax.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
    ax.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    plt.tight_layout()
    p_err_out = os.path.join(PLOTS_DIR, f"06_bidirectional_error_heatmap_{vchar}.png")
    plt.savefig(p_err_out, bbox_inches='tight')
    plt.close()
    print(f"   Карта ошибки сохранена: {p_err_out}")

    # 3. Двухпанельный график: [Карта ошибки + Временные ряды сходимости]
    fig, (ax_map, ax_lines) = plt.subplots(1, 2, figsize=(16.5, 6.8), gridspec_kw={'width_ratios': [1.15, 1]})

    ax_map.set_facecolor('#2c3e50')
    cf_m = ax_map.contourf(K2_grid, K1_grid, log_err_plot, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
    cs_m = ax_map.contour(K2_grid, K1_grid, log_err_plot, levels=[-14, -10, -5, -2, 0], colors='white', alpha=0.5, linewidths=0.8)
    ax_map.clabel(cs_m, inline=True, fontsize=8, fmt=r'$10^{%d}$')
    cbar_m = fig.colorbar(cf_m, ax=ax_map, pad=0.03)
    cbar_m.set_label(r'$\log_{10}(\|e\|)$', fontsize=11, fontweight='bold')
    ax_map.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
    ax_map.set_title(f'Влияние коэффициентов на ошибку (сигнал {vname})', fontweight='bold', pad=10)
    ax_map.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
    ax_map.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
    ax_map.set_xlim(0, 2)
    ax_map.set_ylim(0, 2)

    for k1_p, k2_p, lbl, col_p, lstyle in pairs_to_test:
        xm, ym, zm = xm0, ym0, zm0
        xs, ys, zs = xs0, ys0, zs0
        err_t = np.zeros(s_bidir_curves + 1)
        err_t[0] = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
        for s in range(1, s_bidir_curves + 1):
            xm, ym, zm, xs, ys, zs = step_bidir_channel(xm, ym, zm, xs, ys, zs, h, k1_p, k2_p, var_idx)
            e = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
            err_t[s] = max(e, 1e-16)
        
        ax_lines.plot(t_arr_curves, np.log10(err_t), color=col_p, linestyle=lstyle, lw=1.8, label=lbl)

    ax_lines.axhline(log_eps, color=C_PURPLE, linestyle=':', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
    ax_lines.set_title(f'Сходимость ошибки во времени (сигнал {vname})', fontweight='bold', pad=10)
    ax_lines.set_xlabel('Время t, с', fontweight='bold')
    ax_lines.set_ylabel(r'$\log_{10}(\|e(t)\|)$', fontweight='bold')
    ax_lines.set_xlim(0, t_bidir_curves)
    ax_lines.set_ylim(-16.8, 3.0)
    ax_lines.set_yticks(np.arange(-16, 4, 2))
    ax_lines.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, framealpha=0.95, fontsize=9.5)

    plt.subplots_adjust(top=0.92, bottom=0.22, left=0.08, right=0.98, wspace=0.25)
    p_dual_out = os.path.join(PLOTS_DIR, f"06_bidirectional_error_and_curves_{vchar}.png")
    plt.savefig(p_dual_out, bbox_inches='tight')
    plt.close()
    print(f"   Двухпанельный график сохранен: {p_dual_out}")

# =========================================================================
# БЛОК 2: ВЛИЯНИЕ ДЛИНЫ ОКНА WT НА 1 000 000 ИТЕРАЦИЙ (X, Y, Z)
# =========================================================================
print("\n" + "=" * 80)
print("2. РАСЧЕТ ВЛИЯНИЯ ДЛИНЫ ОКНА WT НА 1 000 000 (1M) ИТЕРАЦИЙ ДЛЯ X, Y, Z")
print("=" * 80)

num_iters_1m = 1_000_000
pt_univ = np.array([-1.2862054512135843, -1.1015760138326, 0.025947709050327224])
k_univ = 2.0
windows = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
styles = ['-', '--', '-.', ':', '-', '--']
colors_wt = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

# Логарифмическая сетка выборки точек
log_sample_indices = np.unique(np.concatenate([
    np.arange(1, 501, dtype=np.int64),
    np.geomspace(500, num_iters_1m, 4500).astype(np.int64)
]))

@njit(fastmath=True)
def run_simulation_sampled(window, wt_steps, num_iters, k, var_idx, h, sample_indices):
    xs = window[0, 0] + 0.15
    ys = window[0, 1] - 0.15
    zs = window[0, 2] + 0.10
    
    n_samples = len(sample_indices)
    sampled_errs = np.zeros(n_samples)
    
    sample_ptr = 0
    next_sample_it = sample_indices[sample_ptr]
    consecutive_zeros = 0
    
    for it in range(1, num_iters + 1):
        for step in range(wt_steps):
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], h, k, var_idx)
        for step in range(wt_steps, 0, -1):
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], -h, -k, var_idx)
        
        e = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)
        
        if e < 1e-14:
            consecutive_zeros += 1
        else:
            consecutive_zeros = 0
            
        if it == next_sample_it:
            sampled_errs[sample_ptr] = e
            sample_ptr += 1
            if sample_ptr < n_samples:
                next_sample_it = sample_indices[sample_ptr]
                
        # Если предел достигнут и стабильно держится более 300 итераций
        if consecutive_zeros > 300:
            while sample_ptr < n_samples:
                sampled_errs[sample_ptr] = e
                sample_ptr += 1
            break
            
    return sampled_errs

@njit(fastmath=True)
def sprott_step_coupled(xs, ys, zs, xm, ym, zm, h, k, var_idx):
    tau = 0.5 * h
    cx = k * (xm - xs) if var_idx == 0 else 0.0
    cy = k * (ym - ys) if var_idx == 1 else 0.0
    cz = k * (zm - zs) if var_idx == 2 else 0.0

    xs1 = xs + tau * (ys * zs + cx)
    ys1 = ys + tau * (xs1 - ys + cy)
    zs1 = zs + tau * (1.0 - xs1 * ys1 + cz)

    zs2 = zs1 + tau * (1.0 - xs1 * ys1 + cz)
    ys2 = (ys1 + tau * (xs1 + cy)) / (1.0 + tau)
    xs2 = xs1 + tau * (ys2 * zs2 + cx)

    return xs2, ys2, zs2

# Warmup JIT
_dummy_w = np.zeros((10, 3))
_dummy_idx = np.array([1, 2, 3], dtype=np.int64)
run_simulation_sampled(_dummy_w, 9, 3, 2.0, 0, h, _dummy_idx)

plot_x_pos = log_sample_indices[log_sample_indices >= 1]

all_channels_1m_data = {}

for var_idx, vname, vchar in channels_bidir:
    print(f"-> Моделирование 1M итераций для канала {vname}...")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ch_curves = {}
    
    for wt, st, c in zip(windows, styles, colors_wt):
        wt_steps = int(round(wt / h))
        window = np.zeros((wt_steps + 1, 3))
        cx, cy, cz = pt_univ[0], pt_univ[1], pt_univ[2]
        for step in range(wt_steps + 1):
            window[step] = [cx, cy, cz]
            cx, cy, cz = sprott_step_cd(cx, cy, cz, h)

        sampled_e = run_simulation_sampled(window, wt_steps, num_iters_1m, k_univ, var_idx, h, log_sample_indices)
        log_e = np.log10(np.clip(sampled_e, 1e-16, 1e2))
        ch_curves[wt] = log_e
        ax.plot(plot_x_pos, log_e, color=c, linestyle=st, lw=1.8, label=f'$W_T = {wt}$ с')

    all_channels_1m_data[vname] = ch_curves

    ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
    ax.set_title(f'Зависимость сходимости ошибки от длины временного окна $W_T$ (канал {vname}, $K = 2.0$)', fontweight='bold', pad=10)
    ax.set_xlabel('Номер итерации i', fontweight='bold')
    ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(1, num_iters_1m)
    ax.set_ylim(-16.8, 3.0)
    ax.set_yticks(np.arange(-16, 4, 2))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=4, framealpha=0.95, fontsize=10)
    plt.subplots_adjust(top=0.92, bottom=0.22, left=0.09, right=0.97)

    p_1m_out = os.path.join(PLOTS_DIR, f"10_reversible_window_comparison_{vchar}_1m_iters.png")
    plt.savefig(p_1m_out, bbox_inches='tight')
    plt.close()
    print(f"   График 1M итераций канала {vname} сохранен: {p_1m_out}")

# Сводный 3-панельный график на 1 000 000 итераций для X, Y, Z
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6.0))
for ax, (var_idx, vname, vchar) in zip([ax1, ax2, ax3], channels_bidir):
    ch_c = all_channels_1m_data[vname]
    for wt, st, c in zip(windows, styles, colors_wt):
        ax.plot(plot_x_pos, ch_c[wt], color=c, linestyle=st, lw=1.8, label=f'$W_T = {wt}$ с')
    ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.4, label=r'$\varepsilon_{mach}$')
    ax.set_title(f'Канал связи {vname}', fontweight='bold', pad=10)
    ax.set_xlabel('Номер итерации i', fontweight='bold')
    ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(1, num_iters_1m)
    ax.set_ylim(-16.8, 3.0)
    ax.set_yticks(np.arange(-16, 4, 2))
    ax.legend(loc='upper right', framealpha=0.90, ncol=2, fontsize=8.5)

plt.suptitle('Сходимость ошибки от длины окна $W_T$ на $10^6$ итерациях (лог-масштаб, $K = 2.0$)', fontweight='bold', y=0.99, fontsize=14)
plt.tight_layout()
p_3panel_1m = os.path.join(PLOTS_DIR, "10_reversible_window_comparison_all_3_1m_iters.png")
plt.savefig(p_3panel_1m, bbox_inches='tight')
plt.close()
print(f"-> Сводный 3-панельный график 1M итераций сохранен: {p_3panel_1m}")

print("\n" + "=" * 80)
print("ГЕНЕРАЦИЯ ПОЛНОСТЬЮ ЗАВЕРШЕНА!")
print("=" * 80)
