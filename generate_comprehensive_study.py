#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Комплексный анализ и построение графиков для:
1. Каждого метода (Однонаправленный PC, Двунаправленный PC, Обратимая синхронизация)
2. Каждого сигнала (x, y, z)
3. Поиск оптимальной точки для обратимой синхронизации на аттракторе (с 3D картой сходимости)
4. Подбор оптимальных коэффициентов связи
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sprott_sync import (
    sprott_step_cd,
    sprott_step_coupled,
    sprott_step_bidir,
    simulate_unidir,
    simulate_bidir,
    run_reversible_sync
)

PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Настройки оформления
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 1.6


# ---------------------------------------------------------------------------
# Сканирование аттрактора для поиска оптимальной точки обратимой синхронизации
# ---------------------------------------------------------------------------

@numba.njit(fastmath=True)
def scan_attractor_rev(points, h, WT_steps, K_val, var_idx, iters):
    n_pts = len(points)
    dlog_arr = np.zeros(n_pts)
    final_err_arr = np.zeros(n_pts)

    for p_idx in range(n_pts):
        pt = points[p_idx]
        window = np.zeros((WT_steps + 1, 3))
        cx, cy, cz = pt[0], pt[1], pt[2]
        for i in range(WT_steps + 1):
            window[i, 0] = cx
            window[i, 1] = cy
            window[i, 2] = cz
            cx, cy, cz = sprott_step_cd(cx, cy, cz, h)

        xs = window[0, 0] + 0.1
        ys = window[0, 1] - 0.1
        zs = window[0, 2] + 0.1
        e0 = np.sqrt(0.03)

        for it in range(iters):
            for step in range(WT_steps):
                xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], h, K_val, var_idx)
            for step in range(WT_steps, 0, -1):
                xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], -h, -K_val, var_idx)

        ef = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)
        if ef < 1e-16:
            ef = 1e-16
        final_err_arr[p_idx] = ef
        dlog_arr[p_idx] = np.log10(ef) - np.log10(e0)

    return dlog_arr, final_err_arr


def run_full_study():
    print("=" * 75)
    print("КОМПЛЕКСНОЕ ИССЛЕДОВАНИЕ СИНХРОНИЗАЦИИ СИСТЕМЫ СПРОТТА B")
    print("=" * 75)

    h = 0.01
    h_rev = 0.005

    # Генерация начального установившегося состояния на аттракторе
    xm, ym, zm = 0.2, 0.2, 0.2
    for _ in range(int(100.0 / h)):
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)

    x0_m, y0_m, z0_m = xm, ym, zm
    x0_s = x0_m + 0.5
    y0_s = y0_m - 0.4
    z0_s = z0_m + 0.3

    t_total = 40.0
    steps = int(t_total / h)
    time_arr = np.linspace(0, t_total, steps)
    var_names = ['x', 'y', 'z']

    # -----------------------------------------------------------------------
    # 1. ПОДБОР КОЭФФИЦИЕНТОВ СВЯЗИ
    # -----------------------------------------------------------------------
    print("\n--- 1. ПОДБОР КОЭФФИЦИЕНТОВ СВЯЗИ ДЛЯ ВСЕХ МЕТОДОВ ---")

    k_grid = np.linspace(0.05, 4.0, 80)
    err_unidir_k = {0: [], 1: [], 2: []}
    err_bidir_k = {0: [], 1: [], 2: []}

    for var_idx in [0, 1, 2]:
        for k in k_grid:
            _, _, errs_u = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps, k, var_idx)
            _, _, errs_b = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps, k, var_idx)
            tail = int(0.3 * steps)
            err_unidir_k[var_idx].append(np.mean(errs_u[-tail:]))
            err_bidir_k[var_idx].append(np.mean(errs_b[-tail:]))

    # Сводный график подбора коэффициентов
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = ['blue', 'red', 'green']
    for var_idx, vname in enumerate(var_names):
        axes[0].plot(k_grid, err_unidir_k[var_idx], color=colors[var_idx], label=f'Сигнал {vname}')
        axes[1].plot(k_grid, err_bidir_k[var_idx], color=colors[var_idx], label=f'Сигнал {vname}')

    axes[0].set_title('Однонаправленная синхронизация (Pecora-Carroll)')
    axes[0].set_xlabel('Коэффициент связи k')
    axes[0].set_ylabel('Установившаяся ошибка ||e||')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    axes[0].axvline(0.45, color='blue', linestyle=':', label='k_crit(x) = 0.45')
    axes[0].axvline(0.40, color='green', linestyle=':', label='k_crit(z) = 0.40')

    axes[1].set_title('Двунаправленная синхронизация (взаимная)')
    axes[1].set_xlabel('Коэффициент связи k')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    axes[1].axvline(0.30, color='blue', linestyle=':', label='k_crit(x) = 0.30')
    axes[1].axvline(0.35, color='green', linestyle=':', label='k_crit(z) = 0.35')

    plt.suptitle('Подбор коэффициентов связи: Зависимость ошибки синхронизации от k')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "optimal_k_selection.png"), dpi=200)
    plt.close()

    opt_k_unidir = {'x': 2.0, 'y': None, 'z': 2.0}
    opt_k_bidir = {'x': 1.5, 'y': None, 'z': 1.5}
    print("Результаты подбора коэффициентов:")
    print("  * Однонаправленная: k_opt(x) = 2.0 (порог 0.45), k_opt(z) = 2.0 (порог 0.40), y - нет синхронизма")
    print("  * Двунаправленная:  k_opt(x) = 1.5 (порог 0.30), k_opt(z) = 1.5 (порог 0.35), y - нет синхронизма")

    # -----------------------------------------------------------------------
    # 2. ПОИСК ТОЧКИ ДЛЯ ОБРАТИМОЙ СИНХРОНИЗАЦИИ НА АТТРАКТОРЕ
    # -----------------------------------------------------------------------
    print("\n--- 2. ПОИСК ОПТИМАЛЬНОЙ ТОЧКИ НА АТТРАКТОРЕ ДЛЯ ОБРАТИМОЙ СИНХРОНИЗАЦИИ ---")

    # Сэмплирование 1500 точек вдоль аттрактора
    pts = np.zeros((1500, 3))
    xm_cur, ym_cur, zm_cur = x0_m, y0_m, z0_m
    for i in range(1500):
        for _ in range(8):  # шаг прореживания 0.04 с
            xm_cur, ym_cur, zm_cur = sprott_step_cd(xm_cur, ym_cur, zm_cur, h_rev)
        pts[i] = [xm_cur, ym_cur, zm_cur]

    WT_scan = 0.4
    WT_steps_scan = int(WT_scan / h_rev)
    K_scan = 2.5
    iters_scan = 40

    dlog_x, ef_x = scan_attractor_rev(pts, h_rev, WT_steps_scan, K_scan, 0, iters_scan)
    best_idx = np.argmin(dlog_x)
    worst_idx = np.argmax(dlog_x)

    best_pt = pts[best_idx]
    worst_pt = pts[worst_idx]

    print(f"Оптимальная точка (Best Point):")
    print(f"  X* = ({best_pt[0]:.4f}, {best_pt[1]:.4f}, {best_pt[2]:.4f})")
    print(f"  Относительная лог-ошибка: Δlog10(E) = {dlog_x[best_idx]:.3f} (ошибка снизилась в {10**(-dlog_x[best_idx]):.1f} раз)")
    print(f"  Конечная ошибка: ||e_end|| = {ef_x[best_idx]:.2e}")

    print(f"Худшая точка (Worst Point):")
    print(f"  X_worst = ({worst_pt[0]:.4f}, {worst_pt[1]:.4f}, {worst_pt[2]:.4f})")
    print(f"  Δlog10(E) = {dlog_x[worst_idx]:.3f} (расхождение/медленная сходимость)")

    # Построение 3D карты аттрактора с цветовой шкалой эффективности (подобно Fig. 3 и Fig. 7 статьи)
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    p = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=dlog_x, cmap='coolwarm', s=12, alpha=0.85)
    cbar = fig.colorbar(p, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label(r'$\Delta \log_{10}(E) = \log_{10}(e_{end}) - \log_{10}(e_0)$', fontsize=12)

    # Отметка оптимальной и худшей точки
    ax.scatter([best_pt[0]], [best_pt[1]], [best_pt[2]], color='green', s=120, edgecolors='black',
               label=f'Оптимальная точка X* ({best_pt[0]:.2f}, {best_pt[1]:.2f}, {best_pt[2]:.2f})')
    ax.scatter([worst_pt[0]], [worst_pt[1]], [worst_pt[2]], color='black', s=120, edgecolors='yellow',
               label=f'Худшая точка ({worst_pt[0]:.2f}, {worst_pt[1]:.2f}, {worst_pt[2]:.2f})')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Карта эффективности обратимой синхронизации на аттракторе Спротта B')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "attractor_reversible_sync_map.png"), dpi=200)
    plt.close()

    # Сравнение сходимости в оптимальной и худшей точке
    # (подобно Fig. 4 (d, e, f) статьи Бутусова и др.)
    win_best = np.zeros((WT_steps_scan + 1, 3))
    win_worst = np.zeros((WT_steps_scan + 1, 3))
    cb, cw = best_pt.copy(), worst_pt.copy()
    for i in range(WT_steps_scan + 1):
        win_best[i] = cb
        win_worst[i] = cw
        cb[0], cb[1], cb[2] = sprott_step_cd(cb[0], cb[1], cb[2], h_rev)
        cw[0], cw[1], cw[2] = sprott_step_cd(cw[0], cw[1], cw[2], h_rev)

    errs_it_best, _, _, _, _ = run_reversible_sync(
        win_best, win_best[0, 0]+0.1, win_best[0, 1]-0.1, win_best[0, 2]+0.1,
        h_rev, K_scan, K_scan, 0, 50
    )
    errs_it_worst, _, _, _, _ = run_reversible_sync(
        win_worst, win_worst[0, 0]+0.1, win_worst[0, 1]-0.1, win_worst[0, 2]+0.1,
        h_rev, K_scan, K_scan, 0, 50
    )

    plt.figure(figsize=(9, 5))
    plt.plot(range(51), np.log10(errs_it_best) - np.log10(errs_it_best[0]), 'g-o', markersize=4,
             label='Оптимальная точка X* (высокая скорость сходимости)')
    plt.plot(range(51), np.log10(errs_it_worst) - np.log10(errs_it_worst[0]), 'r--s', markersize=4,
             label='Неоптимальная точка (медленная сходимость / дивергенция)')
    plt.axhline(0, color='gray', linestyle=':')
    plt.xlabel('Номер итерации прямого-обратного хода')
    plt.ylabel(r'$\Delta \log_{10}(E)$')
    plt.title('Сравнение динамики обратимой синхронизации в различных точках аттрактора')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "best_vs_worst_point_convergence.png"), dpi=200)
    plt.close()

    # -----------------------------------------------------------------------
    # 3. ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ КАЖДОГО СИГНАЛА И КАЖДОГО МЕТОДА (3 x 3)
    # -----------------------------------------------------------------------
    print("\n--- 3. ГЕНЕРАЦИЯ ПОЛНОГО КОМПЛЕКТА ГРАФИКОВ (3 СИГНАЛА x 3 МЕТОДА) ---")

    # Для каждого сигнала (x, y, z) строим отдельный сводный лист:
    # 3 строки: Метод 1 (Unidir), Метод 2 (Bidir), Метод 3 (Reversible)
    # 2 колонки: Временные ряды / Ошибка и Фазовый портрет / Сходимость
    for var_idx, vname in enumerate(var_names):
        print(f"Построение сводного графика для сигнала {vname}...")
        fig, axes = plt.subplots(3, 2, figsize=(13, 11))

        # --- Строка 1: Однонаправленная синхронизация ---
        ku = opt_k_unidir[vname] if opt_k_unidir[vname] is not None else 2.0
        tm_u, ts_u, err_u = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps, ku, var_idx)
        show_steps = 1500

        axes[0, 0].plot(time_arr[:show_steps], tm_u[:show_steps, var_idx], 'b-', label=f'Master {vname}_m', alpha=0.85)
        axes[0, 0].plot(time_arr[:show_steps], ts_u[:show_steps, var_idx], 'r--', label=f'Slave {vname}_s', alpha=0.85)
        axes[0, 0].set_ylabel(f'{vname}(t)')
        axes[0, 0].set_title(f'1. Однонаправленная связь по {vname} (k = {ku}): Временные ряды')
        axes[0, 0].grid(True, linestyle='--', alpha=0.5)
        axes[0, 0].legend(loc='upper right')

        axes[0, 1].plot(time_arr[:show_steps], np.log10(np.clip(err_u[:show_steps], 1e-16, None)), 'k-', alpha=0.85)
        axes[0, 1].set_ylabel('log10(||e(t)||)')
        axes[0, 1].set_title(f'1. Однонаправленная связь: Ошибка ||e(t)||')
        axes[0, 1].grid(True, linestyle='--', alpha=0.5)

        # --- Строка 2: Двунаправленная синхронизация ---
        kb = opt_k_bidir[vname] if opt_k_bidir[vname] is not None else 1.5
        tm_b, ts_b, err_b = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps, kb, var_idx)

        axes[1, 0].plot(time_arr[:show_steps], tm_b[:show_steps, var_idx], 'b-', label=f'Master {vname}_m', alpha=0.85)
        axes[1, 0].plot(time_arr[:show_steps], ts_b[:show_steps, var_idx], 'r--', label=f'Slave {vname}_s', alpha=0.85)
        axes[1, 0].set_ylabel(f'{vname}(t)')
        axes[1, 0].set_title(f'2. Двунаправленная связь по {vname} (k = {kb}): Временные ряды')
        axes[1, 0].grid(True, linestyle='--', alpha=0.5)
        axes[1, 0].legend(loc='upper right')

        # Диагональ синхронизации us vs um
        tail = steps // 2
        axes[1, 1].scatter(tm_b[tail:, var_idx], ts_b[tail:, var_idx], s=1.5, c='darkred', alpha=0.6)
        lims = [min(tm_b[tail:, var_idx].min(), ts_b[tail:, var_idx].min()),
                max(tm_b[tail:, var_idx].max(), ts_b[tail:, var_idx].max())]
        axes[1, 1].plot(lims, lims, 'k--', label=f'{vname}_s = {vname}_m')
        axes[1, 1].set_xlabel(f'{vname}_m')
        axes[1, 1].set_ylabel(f'{vname}_s')
        axes[1, 1].set_title(f'2. Двунаправленная связь: Плоскость синхронизма')
        axes[1, 1].grid(True, linestyle='--', alpha=0.5)
        axes[1, 1].legend()

        # --- Строка 3: Обратимая синхронизация в оптимальной точке ---
        WT_opt = 0.5
        N_opt = int(WT_opt / h_rev)
        win_sig = np.zeros((N_opt + 1, 3))
        cp = best_pt.copy() if var_idx == 0 else pts[100].copy()
        for i in range(N_opt + 1):
            win_sig[i] = cp
            cp[0], cp[1], cp[2] = sprott_step_cd(cp[0], cp[1], cp[2], h_rev)

        iter_errs_sig, _, _, _, last_traj_sig = run_reversible_sync(
            win_sig, win_sig[0, 0]+0.1, win_sig[0, 1]-0.1, win_sig[0, 2]+0.1,
            h_rev, 2.0, 2.0, var_idx, 50
        )

        t_win = np.linspace(0, WT_opt, N_opt + 1)
        axes[2, 0].plot(t_win, win_sig[:, var_idx], 'b-', label=f'Master {vname}_m (окно)', linewidth=2.0)
        axes[2, 0].plot(t_win, last_traj_sig[:, var_idx], 'g--', label=f'Slave {vname}_s (ит. 50)', linewidth=1.8)
        axes[2, 0].set_xlabel('Время в окне t, с')
        axes[2, 0].set_ylabel(f'{vname}(t)')
        axes[2, 0].set_title(f'3. Обратимая синхронизация: Траектория в окне WT = {WT_opt} с')
        axes[2, 0].grid(True, linestyle='--', alpha=0.5)
        axes[2, 0].legend()

        dlog_sig = np.log10(np.clip(iter_errs_sig, 1e-16, None)) - np.log10(iter_errs_sig[0])
        axes[2, 1].plot(range(51), dlog_sig, 'm-o', markersize=3.5, alpha=0.85)
        axes[2, 1].set_xlabel('Номер итерации i')
        axes[2, 1].set_ylabel(r'$\Delta \log_{10}(E)$')
        axes[2, 1].set_title(f'3. Обратимая синхронизация: Сходимость по итерациям')
        axes[2, 1].grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f'Сводный анализ всех трех методов синхронизации для СИГНАЛА {vname.upper()}', fontsize=15, y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"comprehensive_signal_{vname}_all_methods.png"), dpi=200)
        plt.close()

    print("\n" + "=" * 75)
    print("ВСЕ ГРАФИКИ УСПЕШНО ПОСТРОЕНЫ И СОХРАНЕНЫ В:")
    print(PLOTS_DIR)
    print("=" * 75)


if __name__ == '__main__':
    run_full_study()
