#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Генерация всех тепловых карт ошибки и времени двунаправленной синхронизации
в диапазоне коэффициентов связи K1, K2 in [-0.5, 20.0] для сигналов X, Y, Z.

Создаются:
1. Отдельные карты ошибки для каналов X, Y, Z
2. Отдельные карты времени синхронизации (шкала inferno без контурных линий) для X, Y, Z
3. Двухпанельные графики (тепловая карта ошибки + временные ряды сходимости ошибки)
4. Сводная 6-панельная матрица (X, Y, Z: Ошибка и Время)
5. Сводный 4-панельный график (X и Z)
6. Сводная 3-панельная карта времени
7. Сводная 3-панельная карта ошибки
8. Итоговый PDF-документ
9. Выгрузка в отдельный ZIP-архив
"""

import os
import sys
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from numba import njit, prange

BASE_DIR = '/home/bobbert/Документы/leti/mnds/lab4/lab5'
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
OUT_DIR = os.path.join(PLOTS_DIR, 'heatmaps_neg05_to_20')
os.makedirs(OUT_DIR, exist_ok=True)
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
    'grid.alpha': 0.45,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

h = 0.005

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

@njit(fastmath=True)
def simulate_bidir_trajectory(h, n_steps, k1, k2, var_idx):
    err_traj = np.zeros(n_steps + 1)
    xm, ym, zm = 0.5, -0.2, 0.4
    xs, ys, zs = 1.0, 0.3, 0.7
    err_traj[0] = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
    
    for s in range(1, n_steps + 1):
        xm, ym, zm, xs, ys, zs = step_bidir_channel(xm, ym, zm, xs, ys, zs, h, k1, k2, var_idx)
        if abs(xm) > 1e2 or abs(xs) > 1e2 or np.isnan(xm) or np.isnan(xs):
            for rem in range(s, n_steps + 1):
                err_traj[rem] = np.nan
            break
        err_traj[s] = np.sqrt((xm - xs)**2 + (ym - ys)**2 + (zm - zs)**2)
        
    return err_traj

def main():
    print("=" * 80)
    print("ГЕНЕРАЦИЯ ТЕПЛОВЫХ КАРТ В ДИАПАЗОНЕ [-0.5, 20.0] x [-0.5, 20.0]")
    print("=" * 80)

    # Сетка параметров: от -0.5 до 20.0 с шагом 0.1 (206 точек)
    n_k = 206
    k_vals = np.linspace(-0.5, 20.0, n_k)
    K2_grid, K1_grid = np.meshgrid(k_vals, k_vals)

    channels = [
        (0, 'X', 'x'),
        (1, 'Y', 'y'),
        (2, 'Z', 'z')
    ]

    t_eval_curves = 150.0
    s_curves = int(t_eval_curves / h)
    t_axis = np.linspace(0, t_eval_curves, s_curves + 1)

    pairs_to_test = [
        (0.0, 5.0, r'$k_1 = 0.0,\ k_2 = 5.0$', '#7f7f7f', '-'),
        (1.0, 1.0, r'$k_1 = 1.0,\ k_2 = 1.0$', '#ff7f0e', '--'),
        (5.0, 5.0, r'$k_1 = 5.0,\ k_2 = 5.0$', '#1f77b4', '-'),
        (10.0, 10.0, r'$k_1 = 10.0,\ k_2 = 10.0$', '#2ca02c', '-.'),
        (0.5, 0.5, r'$k_1 = 0.5,\ k_2 = 0.5$', '#d62728', ':'),
        (-0.2, 5.0, r'$k_1 = -0.2,\ k_2 = 5.0$', '#9467bd', '--')
    ]

    maps_data = {}

    for var_idx, vname, vchar in channels:
        print(f"\n-> Расчет карт для канала связи {vname}...")
        s_trans = int(100.0 / h) if var_idx == 2 else int(80.0 / h)
        s_eval = int(20.0 / h)
        thresh = 0.05

        err_m, time_m = compute_bidir_maps(k_vals, k_vals, h, s_trans, s_eval, thresh, var_idx)
        log_err_plot = np.log10(np.clip(err_m.T, 1e-16, 1e2))
        time_plot = time_m.T
        maps_data[vchar] = {
            'err': log_err_plot,
            'time': time_plot,
            'vname': vname
        }

        # -------------------------------------------------------------
        # 1. Отдельная тепловая карта времени (шкала лавы inferno, БЕЗ линий)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.5, 8.0))
        ax.set_facecolor('#1e272e')
        t_max_val = np.nanmax(time_plot) if not np.all(np.isnan(time_plot)) else 80.0
        if not np.isnan(t_max_val) and t_max_val > 2.0:
            cf_t = ax.contourf(K2_grid, K1_grid, time_plot, levels=np.linspace(2, min(t_max_val, 100.0), 120), cmap='inferno', extend='both')
            cbar_t = fig.colorbar(cf_t, ax=ax, pad=0.03, aspect=25)
            cbar_t.set_label(r'Время синхронизации $t_{sync}$, с', fontsize=12, fontweight='bold')
        else:
            ax.text(9.75, 9.75, 'Синхронизация не достигается\n(поперечный показатель Ляпунова $\\lambda_\\perp \\ge 0$)',
                    color='#f39c12', fontsize=12, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#2c3e50', edgecolor='#f39c12'))

        ax.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
        ax.set_title(f'Время достижения синхронизации (канал связи {vname})\n$K_1, K_2 \\in [-0.5, 20.0]$', fontweight='bold', pad=12)
        ax.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
        ax.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
        ax.set_xlim(-0.5, 20.0)
        ax.set_ylim(-0.5, 20.0)
        plt.tight_layout()
        p_time_out = os.path.join(OUT_DIR, f"heatmap_sync_time_{vchar}_range_neg05_20.png")
        plt.savefig(p_time_out, bbox_inches='tight')
        plt.close()
        print(f"   Карта времени сохранена: {p_time_out}")

        # -------------------------------------------------------------
        # 2. Отдельная тепловая карта ошибки
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.5, 8.0))
        ax.set_facecolor('#2c3e50')
        cf_e = ax.contourf(K2_grid, K1_grid, log_err_plot, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
        cs_e = ax.contour(K2_grid, K1_grid, log_err_plot, levels=[-14, -10, -5, -2, 0], colors='white', alpha=0.5, linewidths=0.8)
        ax.clabel(cs_e, inline=True, fontsize=8, fmt=r'$10^{%d}$')
        cbar_e = fig.colorbar(cf_e, ax=ax, pad=0.03, aspect=25)
        cbar_e.set_label(r'$\log_{10}(\|e\|)$', fontsize=12, fontweight='bold')
        ax.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
        ax.set_title(f'Установившаяся ошибка синхронизации (канал связи {vname})\n$K_1, K_2 \\in [-0.5, 20.0]$', fontweight='bold', pad=12)
        ax.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
        ax.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
        ax.set_xlim(-0.5, 20.0)
        ax.set_ylim(-0.5, 20.0)
        plt.tight_layout()
        p_err_out = os.path.join(OUT_DIR, f"heatmap_error_{vchar}_range_neg05_20.png")
        plt.savefig(p_err_out, bbox_inches='tight')
        plt.close()
        print(f"   Карта ошибки сохранена: {p_err_out}")

        # -------------------------------------------------------------
        # 3. Двухпанельный график [Карта ошибки + Временные ряды]
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 6.8), gridspec_kw={'width_ratios': [1.08, 1.0]})
        
        # Левая панель: Тепловая карта ошибки
        ax1.set_facecolor('#2c3e50')
        cf1 = ax1.contourf(K2_grid, K1_grid, log_err_plot, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
        cs1 = ax1.contour(K2_grid, K1_grid, log_err_plot, levels=[-14, -10, -5, -2, 0], colors='white', alpha=0.5, linewidths=0.8)
        ax1.clabel(cs1, inline=True, fontsize=8, fmt=r'$10^{%d}$')
        cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
        cbar1.set_label(r'$\log_{10}(\|e\|)$', fontsize=11, fontweight='bold')
        ax1.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax1.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax1.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
        ax1.set_title(f'А. Установившаяся ошибка в плоскости $(K_2, K_1)$', fontweight='bold', pad=10)
        ax1.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
        ax1.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
        ax1.set_xlim(-0.5, 20.0)
        ax1.set_ylim(-0.5, 20.0)

        # Правая панель: Временные ряды ошибки
        ax2.set_facecolor('white')
        for k1_p, k2_p, lbl, col, ls in pairs_to_test:
            # Для канала Z берем адаптированные пары при необходимости
            if var_idx == 2 and k1_p > 2.0:
                continue
            err_tr = simulate_bidir_trajectory(h, s_curves, k1_p, k2_p, var_idx)
            log_tr = np.log10(np.clip(err_tr, 1e-16, 1e2))
            ax2.plot(t_axis, log_tr, label=lbl, color=col, linestyle=ls, lw=1.7)

        # Для Z добавим резонансные пары в узком окне
        if var_idx == 2:
            z_pairs = [
                (0.4, 0.4, r'$k_1 = 0.4,\ k_2 = 0.4$', '#1f77b4', '-'),
                (0.2, 0.7, r'$k_1 = 0.2,\ k_2 = 0.7$', '#2ca02c', '--'),
                (0.6, 0.5, r'$k_1 = 0.6,\ k_2 = 0.5$', '#ff7f0e', '-.'),
                (1.5, 1.5, r'$k_1 = 1.5,\ k_2 = 1.5$ (срыв)', '#d62728', ':')
            ]
            for k1_p, k2_p, lbl, col, ls in z_pairs:
                err_tr = simulate_bidir_trajectory(h, s_curves, k1_p, k2_p, var_idx)
                log_tr = np.log10(np.clip(err_tr, 1e-16, 1e2))
                ax2.plot(t_axis, log_tr, label=lbl, color=col, linestyle=ls, lw=1.7)

        ax2.axhline(-14, color='gray', linestyle=':', lw=1.0, alpha=0.7, label=r'Предел ($\sim 10^{-14}$)')
        ax2.set_xlabel('Время $t$, с', fontweight='bold')
        ax2.set_ylabel(r'$\log_{10}(\|e(t)\|)$', fontweight='bold')
        ax2.set_title(f'Б. Динамика сходимости ошибки во времени', fontweight='bold', pad=10)
        ax2.set_xlim(0, t_eval_curves)
        ax2.set_ylim(-16, 2)
        ax2.legend(loc='upper right', framealpha=0.9)

        plt.suptitle(f'Двунаправленная синхронизация по сигналу {vname} (диапазон $K_1, K_2 \\in [-0.5, 20.0]$)', fontweight='bold', y=0.98, fontsize=14)
        plt.tight_layout()
        p_dual_out = os.path.join(OUT_DIR, f"heatmap_error_and_curves_{vchar}_range_neg05_20.png")
        plt.savefig(p_dual_out, bbox_inches='tight')
        plt.close()
        print(f"   Двухпанельный график сохранен: {p_dual_out}")

    # =========================================================================
    # 4. СВОДНЫЙ 6-ПАНЕЛЬНЫЙ ГРАФИК (X, Y, Z: ОШИБКА И ВРЕМЯ)
    # =========================================================================
    print("\n-> Построение сводного 6-панельного графика (X, Y, Z)...")
    fig, axes = plt.subplots(3, 2, figsize=(16.8, 21.0))

    ch_list = [('x', 'X', 0), ('y', 'Y', 1), ('z', 'Z', 2)]
    for row, (vchar, vname, vidx) in enumerate(ch_list):
        ax_err = axes[row, 0]
        ax_tm = axes[row, 1]

        log_err_data = maps_data[vchar]['err']
        time_data = maps_data[vchar]['time']

        # Ошибка
        ax_err.set_facecolor('#2c3e50')
        cf_e = ax_err.contourf(K2_grid, K1_grid, log_err_data, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
        ax_err.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax_err.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax_err.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
        ax_err.set_title(f'Канал {vname}: Установившаяся ошибка $\\log_{{10}}(\\|e\\|)$', fontweight='bold', pad=10)
        ax_err.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
        ax_err.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
        ax_err.set_xlim(-0.5, 20.0)
        ax_err.set_ylim(-0.5, 20.0)
        cbar_e = fig.colorbar(cf_e, ax=ax_err, pad=0.02)
        cbar_e.set_label(r'$\log_{10}(\|e\|)$', fontsize=10, fontweight='bold')

        # Время
        ax_tm.set_facecolor('#1e272e')
        t_max_v = np.nanmax(time_data) if not np.all(np.isnan(time_data)) else 80.0
        if not np.isnan(t_max_v) and t_max_v > 2.0:
            cf_t = ax_tm.contourf(K2_grid, K1_grid, time_data, levels=np.linspace(2, min(t_max_v, 100.0), 120), cmap='inferno', extend='both')
            cbar_t = fig.colorbar(cf_t, ax=ax_tm, pad=0.02)
            cbar_t.set_label(r'$t_{sync}$, с', fontsize=10, fontweight='bold')
        else:
            ax_tm.text(9.75, 9.75, 'Синхронизация не достигается\n($\\lambda_\\perp \\ge 0$)',
                       color='#f39c12', fontsize=11, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', edgecolor='#f39c12'))

        ax_tm.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax_tm.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax_tm.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
        ax_tm.set_title(f'Канал {vname}: Время синхронизации $t_{{sync}}$', fontweight='bold', pad=10)
        ax_tm.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
        ax_tm.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
        ax_tm.set_xlim(-0.5, 20.0)
        ax_tm.set_ylim(-0.5, 20.0)

    plt.suptitle('Сравнительный анализ двунаправленной синхронизации по каналам X, Y, Z ($K_1, K_2 \\in [-0.5, 20.0]$)',
                 fontweight='bold', y=0.995, fontsize=15)
    plt.tight_layout()
    p_6panel = os.path.join(OUT_DIR, "heatmap_comparison_6panel_range_neg05_20.png")
    plt.savefig(p_6panel, bbox_inches='tight')
    plt.close()
    print(f"   Сводный 6-панельный график сохранен: {p_6panel}")

    # =========================================================================
    # 5. СВОДНЫЙ 4-ПАНЕЛЬНЫЙ ГРАФИК (X И Z: ОШИБКА И ВРЕМЯ)
    # =========================================================================
    print("-> Построение сводного 4-панельного графика (X и Z)...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16.8, 14.5))

    # X - Ошибка
    ax1.set_facecolor('#2c3e50')
    cf1 = ax1.contourf(K2_grid, K1_grid, maps_data['x']['err'], levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
    ax1.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
    ax1.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
    ax1.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
    ax1.set_title('А. Установившаяся ошибка (канал связи X)', fontweight='bold', pad=10)
    ax1.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
    ax1.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
    ax1.set_xlim(-0.5, 20.0)
    ax1.set_ylim(-0.5, 20.0)
    cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
    cbar1.set_label(r'$\log_{10}(\|e\|)$', fontsize=11, fontweight='bold')

    # X - Время
    ax2.set_facecolor('#1e272e')
    cf2 = ax2.contourf(K2_grid, K1_grid, maps_data['x']['time'], levels=np.linspace(2, 85, 120), cmap='inferno', extend='both')
    ax2.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
    ax2.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
    ax2.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
    ax2.set_title('Б. Время синхронизации $t_{sync}$ (канал связи X)', fontweight='bold', pad=10)
    ax2.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
    ax2.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
    ax2.set_xlim(-0.5, 20.0)
    ax2.set_ylim(-0.5, 20.0)
    cbar2 = fig.colorbar(cf2, ax=ax2, pad=0.03)
    cbar2.set_label(r'$t_{sync}$, с', fontsize=11, fontweight='bold')

    # Z - Ошибка
    ax3.set_facecolor('#2c3e50')
    cf3 = ax3.contourf(K2_grid, K1_grid, maps_data['z']['err'], levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
    ax3.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
    ax3.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
    ax3.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
    ax3.set_title('В. Установившаяся ошибка (канал связи Z)', fontweight='bold', pad=10)
    ax3.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
    ax3.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
    ax3.set_xlim(-0.5, 20.0)
    ax3.set_ylim(-0.5, 20.0)
    cbar3 = fig.colorbar(cf3, ax=ax3, pad=0.03)
    cbar3.set_label(r'$\log_{10}(\|e\|)$', fontsize=11, fontweight='bold')

    # Z - Время
    ax4.set_facecolor('#1e272e')
    cf4 = ax4.contourf(K2_grid, K1_grid, maps_data['z']['time'], levels=np.linspace(2, 90, 120), cmap='inferno', extend='both')
    ax4.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
    ax4.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
    ax4.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
    ax4.set_title('Г. Время синхронизации $t_{sync}$ (канал связи Z)', fontweight='bold', pad=10)
    ax4.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
    ax4.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
    ax4.set_xlim(-0.5, 20.0)
    ax4.set_ylim(-0.5, 20.0)
    cbar4 = fig.colorbar(cf4, ax=ax4, pad=0.03)
    cbar4.set_label(r'$t_{sync}$, с', fontsize=11, fontweight='bold')

    plt.suptitle('Сравнительный анализ каналов X и Z ($K_1, K_2 \\in [-0.5, 20.0]$)', fontweight='bold', y=0.99, fontsize=15)
    plt.tight_layout()
    p_4panel = os.path.join(OUT_DIR, "heatmap_comparison_4panel_xz_range_neg05_20.png")
    plt.savefig(p_4panel, bbox_inches='tight')
    plt.close()
    print(f"   Сводный 4-панельный график сохранен: {p_4panel}")

    # =========================================================================
    # 6. СВОДНАЯ 3-ПАНЕЛЬНАЯ КАРТА ВРЕМЕНИ ДЛЯ X, Y, Z
    # =========================================================================
    print("-> Построение сводной 3-панельной карты времени...")
    fig, axes = plt.subplots(1, 3, figsize=(22.0, 6.8))
    for i, (vchar, vname, vidx) in enumerate(ch_list):
        ax = axes[i]
        ax.set_facecolor('#1e272e')
        time_data = maps_data[vchar]['time']
        t_max_v = np.nanmax(time_data) if not np.all(np.isnan(time_data)) else 80.0
        if not np.isnan(t_max_v) and t_max_v > 2.0:
            cf = ax.contourf(K2_grid, K1_grid, time_data, levels=np.linspace(2, min(t_max_v, 100.0), 120), cmap='inferno', extend='both')
            cbar = fig.colorbar(cf, ax=ax, pad=0.03)
            cbar.set_label(r'$t_{sync}$, с', fontsize=10, fontweight='bold')
        else:
            ax.text(9.75, 9.75, 'Синхронизация не достигается\n($\\lambda_\\perp \\ge 0$)',
                    color='#f39c12', fontsize=11, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', edgecolor='#f39c12'))
        ax.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
        ax.set_title(f'Канал {vname}: Время $t_{{sync}}$', fontweight='bold', pad=10)
        ax.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
        ax.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
        ax.set_xlim(-0.5, 20.0)
        ax.set_ylim(-0.5, 20.0)

    plt.suptitle('Карты времени синхронизации $t_{sync}$ для сигналов X, Y, Z в плоскости $K_1, K_2 \\in [-0.5, 20.0]$',
                 fontweight='bold', y=0.98, fontsize=14)
    plt.tight_layout()
    p_3time = os.path.join(OUT_DIR, "heatmap_sync_time_xyz_triptych_range_neg05_20.png")
    plt.savefig(p_3time, bbox_inches='tight')
    plt.close()
    print(f"   Сводный триптих времени сохранен: {p_3time}")

    # =========================================================================
    # 7. СВОДНАЯ 3-ПАНЕЛЬНАЯ КАРТА ОШИБКИ ДЛЯ X, Y, Z
    # =========================================================================
    print("-> Построение сводной 3-панельной карты ошибки...")
    fig, axes = plt.subplots(1, 3, figsize=(22.0, 6.8))
    for i, (vchar, vname, vidx) in enumerate(ch_list):
        ax = axes[i]
        ax.set_facecolor('#2c3e50')
        err_data = maps_data[vchar]['err']
        cf = ax.contourf(K2_grid, K1_grid, err_data, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
        cbar = fig.colorbar(cf, ax=ax, pad=0.03)
        cbar.set_label(r'$\log_{10}(\|e\|)$', fontsize=10, fontweight='bold')
        ax.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.5)
        ax.plot([-0.5, 20], [-0.5, 20], color='white', linestyle='--', lw=1.2, alpha=0.7)
        ax.set_title(f'Канал {vname}: Ошибка $\\log_{{10}}(\\|e\\|)$', fontweight='bold', pad=10)
        ax.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
        ax.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
        ax.set_xlim(-0.5, 20.0)
        ax.set_ylim(-0.5, 20.0)

    plt.suptitle('Карты установившейся ошибки $\\log_{10}(\\|e\\|)$ для сигналов X, Y, Z в плоскости $K_1, K_2 \\in [-0.5, 20.0]$',
                 fontweight='bold', y=0.98, fontsize=14)
    plt.tight_layout()
    p_3err = os.path.join(OUT_DIR, "heatmap_error_xyz_triptych_range_neg05_20.png")
    plt.savefig(p_3err, bbox_inches='tight')
    plt.close()
    print(f"   Сводный триптих ошибки сохранен: {p_3err}")

    # =========================================================================
    # 8. ГЕНЕРАЦИЯ ЕДИНОГО PDF ОТЧЕТА СО ВСЕМИ КАРТАМИ
    # =========================================================================
    pdf_out = os.path.join(OUT_DIR, "heatmaps_neg05_to_20_report.pdf")
    print(f"\n-> Создание многостраничного PDF: {pdf_out}...")
    from PIL import Image

    pdf_image_files = [
        os.path.join(OUT_DIR, "heatmap_comparison_6panel_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_comparison_4panel_xz_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_sync_time_xyz_triptych_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_error_xyz_triptych_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_error_and_curves_x_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_sync_time_x_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_error_x_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_error_and_curves_y_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_sync_time_y_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_error_y_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_error_and_curves_z_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_sync_time_z_range_neg05_20.png"),
        os.path.join(OUT_DIR, "heatmap_error_z_range_neg05_20.png"),
    ]

    with PdfPages(pdf_out) as pdf:
        for img_path in pdf_image_files:
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path)
            w, h_img = img.size
            fig_w = 12.0
            fig_h = 12.0 * (h_img / w)
            fig_pdf = plt.figure(figsize=(fig_w, fig_h), dpi=150)
            ax_pdf = fig_pdf.add_axes([0, 0, 1, 1])
            ax_pdf.imshow(img)
            ax_pdf.axis('off')
            pdf.savefig(fig_pdf, dpi=200)
            plt.close(fig_pdf)
    print(f"   PDF отчет успешно создан: {pdf_out}")

    # Также копируем ключевые графики в основную папку plots для обратной совместимости
    for src_name in [
        "heatmap_comparison_6panel_range_neg05_20.png",
        "heatmap_comparison_4panel_xz_range_neg05_20.png",
        "heatmap_sync_time_x_range_neg05_20.png",
        "heatmap_sync_time_y_range_neg05_20.png",
        "heatmap_sync_time_z_range_neg05_20.png",
        "heatmap_error_x_range_neg05_20.png",
        "heatmap_error_y_range_neg05_20.png",
        "heatmap_error_z_range_neg05_20.png",
        "heatmaps_neg05_to_20_report.pdf"
    ]:
        src_file = os.path.join(OUT_DIR, src_name)
        dst_file = os.path.join(PLOTS_DIR, src_name)
        if os.path.exists(src_file):
            import shutil
            shutil.copy2(src_file, dst_file)

    # =========================================================================
    # 9. СОЗДАНИЕ ОТДЕЛЬНОГО ZIP-АРХИВА СО ВСЕМИ НОВЫМИ КАРТАМИ
    # =========================================================================
    zip_out = os.path.join(BASE_DIR, "heatmaps_neg05_to_20.zip")
    print(f"\n-> Упаковка всех сгенерированных карт в отдельный архив {zip_out}...")
    with zipfile.ZipFile(zip_out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(OUT_DIR):
            f_path = os.path.join(OUT_DIR, f)
            if os.path.isfile(f_path):
                zf.write(f_path, arcname=f)
    print(f"   Архив успешно создан: {zip_out} (размер: {os.path.getsize(zip_out) / (1024*1024):.2f} МБ)")

    print("\n" + "=" * 80)
    print("ВСЕ РАСЧЕТЫ И ГЕНЕРАЦИЯ УСПЕШНО ЗАВЕРШЕНЫ!")
    print("=" * 80)

if __name__ == '__main__':
    main()
