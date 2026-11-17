#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт построения абсолютно чистых тепловых карт и графиков сходимости по поколениям
для 4 ключевых точек обратимой синхронизации:
1. Точка X*_X (канал X) -> 16_point_x_heatmap_and_generations.png
2. Точка X*_Y (канал Y) -> 17_point_y_heatmap_and_generations.png
3. Точка X*_Z (канал Z) -> 18_point_z_heatmap_and_generations.png
4. Универсальная точка X*_univ (все 3 канала) -> 19_point_univ_heatmap_and_generations.png
5. Сводная матрица тепловых карт 2x2 -> 20_all_points_heatmaps_comparison.png

БЕЗ каких-либо текстовых плашек, выносок, стрелок на графиках!
Кривые сходимости рассчитываются строго по точной динамике (без искусственного зануления и без пересечения 0).
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit, prange

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
sys.path.insert(0, CURRENT_DIR)

from sprott_sync import sprott_step_cd, sprott_step_coupled

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.6,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

@njit(fastmath=True)
def numba_sprott_step_cd(x, y, z, h):
    k1_x = y * z
    k1_y = x - y
    k1_z = 1.0 - x * y

    x_mid = x + 0.5 * h * k1_x
    y_mid = y + 0.5 * h * k1_y
    z_mid = z + 0.5 * h * k1_z

    k2_x = y_mid * z_mid
    k2_y = x_mid - y_mid
    k2_z = 1.0 - x_mid * y_mid

    return x + h * k2_x, y + h * k2_y, z + h * k2_z

@njit(fastmath=True)
def numba_sprott_step_coupled(xs, ys, zs, xm, ym, zm, h, k, var_idx):
    diff_x = xm - xs if var_idx == 0 else 0.0
    diff_y = ym - ys if var_idx == 1 else 0.0
    diff_z = zm - zs if var_idx == 2 else 0.0

    k1_x = ys * zs + k * diff_x
    k1_y = xs - ys + k * diff_y
    k1_z = 1.0 - xs * ys + k * diff_z

    xs_mid = xs + 0.5 * h * k1_x
    ys_mid = ys + 0.5 * h * k1_y
    zs_mid = zs + 0.5 * h * k1_z

    diff_x_mid = xm - xs_mid if var_idx == 0 else 0.0
    diff_y_mid = ym - ys_mid if var_idx == 1 else 0.0
    diff_z_mid = zm - zs_mid if var_idx == 2 else 0.0

    k2_x = ys_mid * zs_mid + k * diff_x_mid
    k2_y = xs_mid - ys_mid + k * diff_y_mid
    k2_z = 1.0 - xs_mid * ys_mid + k * diff_z_mid

    return xs + h * k2_x, ys + h * k2_y, zs + h * k2_z

@njit(parallel=True, fastmath=True)
def compute_pt_grid(pt, var_idx, wt_vals, k_vals, iters=25):
    n_wt = len(wt_vals)
    n_k = len(k_vals)
    grid_res = np.zeros((n_k, n_wt))
    h = 0.005

    for iw in prange(n_wt):
        wt = wt_vals[iw]
        wt_steps = int(round(wt / h))

        window = np.zeros((wt_steps + 1, 3))
        cx, cy, cz = pt[0], pt[1], pt[2]
        for s in range(wt_steps + 1):
            window[s, 0] = cx
            window[s, 1] = cy
            window[s, 2] = cz
            cx, cy, cz = numba_sprott_step_cd(cx, cy, cz, h)

        for ik in range(n_k):
            k_val = k_vals[ik]
            xs = window[0, 0] + 0.15
            ys = window[0, 1] - 0.15
            zs = window[0, 2] + 0.10

            err = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)

            for it in range(1, iters + 1):
                for step in range(wt_steps):
                    xs, ys, zs = numba_sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], h, k_val, var_idx)
                for step in range(wt_steps, 0, -1):
                    xs, ys, zs = numba_sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], -h, -k_val, var_idx)

                err = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)

            grid_res[ik, iw] = err

    return grid_res

def run_exact_simulation(pt, wt, k, var_idx, num_iters=30):
    h = 0.005
    wt_steps = int(round(wt / h))
    window = np.zeros((wt_steps + 1, 3))
    cx, cy, cz = pt[0], pt[1], pt[2]
    for step in range(wt_steps + 1):
        window[step] = [cx, cy, cz]
        cx, cy, cz = sprott_step_cd(cx, cy, cz, h)

    xs = window[0, 0] + 0.15
    ys = window[0, 1] - 0.15
    zs = window[0, 2] + 0.10

    errs = np.zeros(num_iters + 1)
    errs[0] = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)

    for it in range(1, num_iters + 1):
        for step in range(wt_steps):
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], h, k, var_idx)
        for step in range(wt_steps, 0, -1):
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], -h, -k, var_idx)
        errs[it] = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)

    return errs

def generate_all_heatmaps_and_plots():
    print("=" * 80)
    print("ГЕНЕРАЦИЯ ЧИСТЫХ ТЕПЛОВЫХ КАРТ И ГРАФИКОВ СХОДИМОСТИ ДЛЯ 4 ТОЧЕК")
    print("=" * 80)

    db_path = os.path.join(CURRENT_DIR, "optimal_points_database.json")
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    rec_x = db['records_by_channel']['X']
    rec_y = db['records_by_channel']['Y']
    rec_z = db['records_by_channel']['Z']
    univ = db['universal_point']

    wt_vals = np.linspace(0.5, 4.0, 71)
    k_vals = np.linspace(0.5, 3.5, 61)
    WT_grid, K_grid = np.meshgrid(wt_vals, k_vals)

    # Прогрев Numba
    _ = compute_pt_grid(np.array(rec_x['best_pt']), 0, wt_vals[:3], k_vals[:3], 3)

    eps_mach = 2.220446049250313e-16
    log_eps = np.log10(eps_mach)

    # -----------------------------------------------------------------------
    # 1. ТОЧКА X*_X (КАНАЛ X) -> Рис. 16
    # -----------------------------------------------------------------------
    print("\n1. Расчет тепловой карты для точки X*_X (канал связи X)...")
    pt_x = np.array(rec_x['best_pt'])
    grid_x = compute_pt_grid(pt_x, 0, wt_vals, k_vals, 25)
    log_grid_x = np.log10(np.clip(grid_x, 1e-16, 1e2))
    errs_x = run_exact_simulation(pt_x, rec_x['best_wt'], rec_x['best_k'], 0, 30)
    log_errs_x = np.log10(errs_x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.1, 1]})

    cf1 = ax1.contourf(WT_grid, K_grid, log_grid_x, levels=np.linspace(-16, 0, 33), cmap='viridis_r', extend='both')
    cs1 = ax1.contour(WT_grid, K_grid, log_grid_x, levels=[-15, -12, -8, -4, -1], colors='white', alpha=0.5, linewidths=0.8)
    ax1.clabel(cs1, inline=True, fontsize=8, fmt=r'$10^{%d}$')
    cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
    cbar1.set_label(r'$\log_{10}(\|e\|)$ (установившаяся ошибка)', fontsize=12)

    # Оптимальная точка (только звездочка, без текстовых выносок)
    ax1.plot(rec_x['best_wt'], rec_x['best_k'], marker='*', markersize=18, color='red', markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax1.set_title(f"Тепловая карта бассейна сходимости для $X^*_X$\n"
                  f"Точка: ({pt_x[0]:.2f}, {pt_x[1]:.2f}, {pt_x[2]:.2f}), Канал $X$", fontsize=12, pad=10)
    ax1.set_xlabel(r"Длительность окна наблюдения $W_T$ (с)", fontsize=11)
    ax1.set_ylabel(r"Коэффициент связи $K$", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4, color='white')

    # График по поколениям (чистый, без выносок)
    generations = np.arange(len(errs_x))
    ax2.plot(generations, log_errs_x, 'o-', color='#1f77b4', lw=2.2, markersize=5, label=r'Ошибка $\|e_i\|$ (канал $X$)')
    ax2.axhline(log_eps, color='#8e44ad', linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach} \approx 2.22 \cdot 10^{-16}$')

    ax2.set_title(f"Сходимость ошибки по циклам (итерациям) для точки $X^*_X$\n$W_T = {rec_x['best_wt']}$ с, $K = {rec_x['best_k']}$", fontsize=12, pad=10)
    ax2.set_xlabel("Номер итерации (цикла прямого-обратного хода) $i$", fontsize=11)
    ax2.set_ylabel(r"$\log_{10}(\|e\|)$", fontsize=11)
    ax2.set_xlim(-0.5, 30.5)
    ax2.set_ylim(-16.8, 0.5)
    ax2.set_yticks(np.arange(-16, 2, 2))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    p16 = os.path.join(PLOTS_DIR, "16_point_x_heatmap_and_generations.png")
    plt.savefig(p16, dpi=300)
    plt.close()
    print(f"-> Сохранено: {p16}")

    # -----------------------------------------------------------------------
    # 2. ТОЧКА X*_Y (КАНАЛ Y) -> Рис. 17
    # -----------------------------------------------------------------------
    print("\n2. Расчет тепловой карты для точки X*_Y (канал связи Y)...")
    pt_y = np.array(rec_y['best_pt'])
    grid_y = compute_pt_grid(pt_y, 1, wt_vals, k_vals, 25)
    log_grid_y = np.log10(np.clip(grid_y, 1e-16, 1e2))
    errs_y = run_exact_simulation(pt_y, rec_y['best_wt'], rec_y['best_k'], 1, 30)
    log_errs_y = np.log10(errs_y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.1, 1]})

    cf1 = ax1.contourf(WT_grid, K_grid, log_grid_y, levels=np.linspace(-16, 1, 35), cmap='magma_r', extend='both')
    cs1 = ax1.contour(WT_grid, K_grid, log_grid_y, levels=[-15, -12, -8, -4, 0], colors='white', alpha=0.5, linewidths=0.8)
    ax1.clabel(cs1, inline=True, fontsize=8, fmt=r'$10^{%d}$')
    cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
    cbar1.set_label(r'$\log_{10}(\|e\|)$ (установившаяся ошибка)', fontsize=12)

    ax1.plot(rec_y['best_wt'], rec_y['best_k'], marker='*', markersize=18, color='cyan', markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax1.set_title(f"Тепловая карта бассейна сходимости для $X^*_Y$\n"
                  f"Точка: ({pt_y[0]:.2f}, {pt_y[1]:.2f}, {pt_y[2]:.2f}), Канал $Y$", fontsize=12, pad=10)
    ax1.set_xlabel(r"Длительность окна наблюдения $W_T$ (с)", fontsize=11)
    ax1.set_ylabel(r"Коэффициент связи $K$", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4, color='white')

    ax2.plot(generations, log_errs_y, 's-', color='#d62728', lw=2.2, markersize=5, label=r'Ошибка $\|e_i\|$ (канал $Y$)')
    ax2.axhline(log_eps, color='#8e44ad', linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach} \approx 2.22 \cdot 10^{-16}$')

    ax2.set_title(f"Сходимость ошибки по циклам (итерациям) для точки $X^*_Y$\n$W_T = {rec_y['best_wt']}$ с, $K = {rec_y['best_k']}$", fontsize=12, pad=10)
    ax2.set_xlabel("Номер итерации (цикла прямого-обратного хода) $i$", fontsize=11)
    ax2.set_ylabel(r"$\log_{10}(\|e\|)$", fontsize=11)
    ax2.set_xlim(-0.5, 30.5)
    ax2.set_ylim(-16.8, 0.5)
    ax2.set_yticks(np.arange(-16, 2, 2))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    p17 = os.path.join(PLOTS_DIR, "17_point_y_heatmap_and_generations.png")
    plt.savefig(p17, dpi=300)
    plt.close()
    print(f"-> Сохранено: {p17}")

    # -----------------------------------------------------------------------
    # 3. ТОЧКА X*_Z (КАНАЛ Z) -> Рис. 18
    # -----------------------------------------------------------------------
    print("\n3. Расчет тепловой карты для точки X*_Z (канал связи Z)...")
    pt_z = np.array(rec_z['best_pt'])
    grid_z = compute_pt_grid(pt_z, 2, wt_vals, k_vals, 25)
    log_grid_z = np.log10(np.clip(grid_z, 1e-16, 1e2))
    errs_z = run_exact_simulation(pt_z, rec_z['best_wt'], rec_z['best_k'], 2, 30)
    log_errs_z = np.log10(errs_z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.1, 1]})

    cf1 = ax1.contourf(WT_grid, K_grid, log_grid_z, levels=np.linspace(-16, 0, 33), cmap='plasma_r', extend='both')
    cs1 = ax1.contour(WT_grid, K_grid, log_grid_z, levels=[-15, -12, -8, -4, -1], colors='white', alpha=0.5, linewidths=0.8)
    ax1.clabel(cs1, inline=True, fontsize=8, fmt=r'$10^{%d}$')
    cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
    cbar1.set_label(r'$\log_{10}(\|e\|)$ (установившаяся ошибка)', fontsize=12)

    ax1.plot(rec_z['best_wt'], rec_z['best_k'], marker='*', markersize=18, color='lime', markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax1.set_title(f"Тепловая карта бассейна сходимости для $X^*_Z$\n"
                  f"Точка: ({pt_z[0]:.2f}, {pt_z[1]:.2f}, {pt_z[2]:.2f}), Канал $Z$", fontsize=12, pad=10)
    ax1.set_xlabel(r"Длительность окна наблюдения $W_T$ (с)", fontsize=11)
    ax1.set_ylabel(r"Коэффициент связи $K$", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4, color='white')

    ax2.plot(generations, log_errs_z, '^-', color='#2ca02c', lw=2.2, markersize=5, label=r'Ошибка $\|e_i\|$ (канал $Z$)')
    ax2.axhline(log_eps, color='#8e44ad', linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach} \approx 2.22 \cdot 10^{-16}$')

    ax2.set_title(f"Сходимость ошибки по циклам (итерациям) для точки $X^*_Z$\n$W_T = {rec_z['best_wt']}$ с, $K = {rec_z['best_k']}$", fontsize=12, pad=10)
    ax2.set_xlabel("Номер итерации (цикла прямого-обратного хода) $i$", fontsize=11)
    ax2.set_ylabel(r"$\log_{10}(\|e\|)$", fontsize=11)
    ax2.set_xlim(-0.5, 30.5)
    ax2.set_ylim(-16.8, 0.5)
    ax2.set_yticks(np.arange(-16, 2, 2))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    p18 = os.path.join(PLOTS_DIR, "18_point_z_heatmap_and_generations.png")
    plt.savefig(p18, dpi=300)
    plt.close()
    print(f"-> Сохранено: {p18}")

    # -----------------------------------------------------------------------
    # 4. УНИВЕРСАЛЬНАЯ ТОЧКА X*_univ (ОДНОВРЕМЕННО X, Y, Z) -> Рис. 19
    # -----------------------------------------------------------------------
    print("\n4. Расчет тепловой карты для универсальной точки X*_univ...")
    pt_u = np.array(univ['coordinates'])
    gx_u = compute_pt_grid(pt_u, 0, wt_vals, k_vals, 25)
    gy_u = compute_pt_grid(pt_u, 1, wt_vals, k_vals, 25)
    gz_u = compute_pt_grid(pt_u, 2, wt_vals, k_vals, 25)
    g_univ = np.maximum(np.maximum(gx_u, gy_u), gz_u)
    log_grid_u = np.log10(np.clip(g_univ, 1e-16, 1e2))

    errs_u_x = run_exact_simulation(pt_u, univ['wt'], univ['k'], 0, 30)
    errs_u_y = run_exact_simulation(pt_u, univ['wt'], univ['k'], 1, 30)
    errs_u_z = run_exact_simulation(pt_u, univ['wt'], univ['k'], 2, 30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.1, 1]})

    cf1 = ax1.contourf(WT_grid, K_grid, log_grid_u, levels=np.linspace(-15, 1, 33), cmap='coolwarm_r', extend='both')
    cs1 = ax1.contour(WT_grid, K_grid, log_grid_u, levels=[-13, -10, -6, -2, 0], colors='black', alpha=0.4, linewidths=0.8)
    ax1.clabel(cs1, inline=True, fontsize=8, fmt=r'$10^{%d}$')
    cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
    cbar1.set_label(r'$\log_{10}(\max(e_x, e_y, e_z))$', fontsize=12)

    ax1.plot(univ['wt'], univ['k'], marker='*', markersize=20, color='gold', markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax1.set_title(f"Тепловая карта универсального оптимума $X^*_{{univ}}$\n"
                  f"Точка: ({pt_u[0]:.2f}, {pt_u[1]:.2f}, {pt_u[2]:.2f}), Худший из трех каналов", fontsize=12, pad=10)
    ax1.set_xlabel(r"Длительность окна наблюдения $W_T$ (с)", fontsize=11)
    ax1.set_ylabel(r"Коэффициент связи $K$", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4, color='black')

    log_ux = np.log10(errs_u_x)
    log_uy = np.log10(errs_u_y)
    log_uz = np.log10(errs_u_z)

    ax2.plot(generations, log_ux, 'o-', color='#1f77b4', lw=2.0, markersize=4, label='Канал связи X')
    ax2.plot(generations, log_uy, 's-', color='#d62728', lw=2.0, markersize=4, linestyle='--', label='Канал связи Y')
    ax2.plot(generations, log_uz, '^-', color='#2ca02c', lw=2.0, markersize=4, linestyle='-.', label='Канал связи Z')
    ax2.axhline(log_eps, color='#8e44ad', linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')

    ax2.set_title(f"Сходимость ошибки по циклам (итерациям) во всех 3 каналах\nУниверсальная точка $X^*_{{univ}}$, $W_T = {univ['wt']}$ с, $K = {univ['k']}$", fontsize=12, pad=10)
    ax2.set_xlabel("Номер итерации (цикла прямого-обратного хода) $i$", fontsize=11)
    ax2.set_ylabel(r"$\log_{10}(\|e\|)$", fontsize=11)
    ax2.set_xlim(-0.5, 30.5)
    ax2.set_ylim(-16.8, 0.5)
    ax2.set_yticks(np.arange(-16, 2, 2))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    p19 = os.path.join(PLOTS_DIR, "19_point_univ_heatmap_and_generations.png")
    plt.savefig(p19, dpi=300)
    plt.close()
    print(f"-> Сохранено: {p19}")

    # -----------------------------------------------------------------------
    # 5. СВОДНАЯ СРАВНИТЕЛЬНАЯ КАРТА 4 ТЕПЛОВЫХ КАРТ (2x2) -> Рис. 20
    # -----------------------------------------------------------------------
    print("\n5. Построение сводной матрицы 2x2 тепловых карт для всех 4 точек...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    maps_data = [
        (axes[0, 0], log_grid_x, f"А. Точка $X^*_X$ (Канал $X$)\n$W_T={rec_x['best_wt']}$ с, $K={rec_x['best_k']}$", rec_x['best_wt'], rec_x['best_k'], 'red', 'viridis_r'),
        (axes[0, 1], log_grid_y, f"Б. Точка $X^*_Y$ (Канал $Y$)\n$W_T={rec_y['best_wt']}$ с, $K={rec_y['best_k']}$", rec_y['best_wt'], rec_y['best_k'], 'cyan', 'magma_r'),
        (axes[1, 0], log_grid_z, f"В. Точка $X^*_Z$ (Канал $Z$)\n$W_T={rec_z['best_wt']}$ с, $K={rec_z['best_k']}$", rec_z['best_wt'], rec_z['best_k'], 'lime', 'plasma_r'),
        (axes[1, 1], log_grid_u, f"Г. Универсальная точка $X^*_{{univ}}$ (Каналы $X,Y,Z$)\n$W_T={univ['wt']}$ с, $K={univ['k']}$", univ['wt'], univ['k'], 'gold', 'coolwarm_r')
    ]

    for ax, g_data, title, b_wt, b_k, star_c, cmp in maps_data:
        cf = ax.contourf(WT_grid, K_grid, g_data, levels=np.linspace(-16, 1, 35), cmap=cmp, extend='both')
        cs = ax.contour(WT_grid, K_grid, g_data, levels=[-15, -10, -5, 0], colors='white', alpha=0.4, linewidths=0.7)
        cbar = fig.colorbar(cf, ax=ax, pad=0.02)
        cbar.set_label(r'$\log_{10}(\|e\|)$', fontsize=11)
        ax.plot(b_wt, b_k, marker='*', markersize=18, color=star_c, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel(r"Окно наблюдения $W_T$ (с)", fontsize=10)
        ax.set_ylabel(r"Коэффициент связи $K$", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.3, color='white')

    plt.suptitle("Сравнительный анализ бассейнов сходимости обратимой синхронизации\nдля четырех ключевых точек в плоскости параметров $(W_T, K)$", fontsize=14, y=0.99, fontweight='bold')
    plt.tight_layout()
    p20 = os.path.join(PLOTS_DIR, "20_all_points_heatmaps_comparison.png")
    plt.savefig(p20, dpi=300)
    plt.close()
    print(f"-> Сохранено: {p20}")

    print("\nВСЕ ТЕПЛОВЫЕ КАРТЫ И ГРАФИКИ УСПЕШНО СФОРМИРОВАНЫ БЕЗ ВЫНОСОК!")

if __name__ == "__main__":
    generate_all_heatmaps_and_plots()
