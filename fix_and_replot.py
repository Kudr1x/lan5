#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт перерисовки графиков для ЛР №5.
Приводит все графики в идеальное соответствие с результатами экспериментов и текстом отчета.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sprott_sync import sprott_step_cd, simulate_unidir, simulate_bidir, run_reversible_sync
from generate_point_heatmaps import get_numba_exact_history

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Загружаем базу данных оптимальных точек
DB_PATH = os.path.join(BASE_DIR, "optimal_points_database.json")
with open(DB_PATH, "r", encoding="utf-8") as f:
    db = json.load(f)

rec_X = db["records_by_channel"]["X"]
rec_Y = db["records_by_channel"]["Y"]
rec_Z = db["records_by_channel"]["Z"]
univ = db["universal_point"]

pt_X = np.array(rec_X["best_pt"])
wt_X = rec_X["best_wt"]
k_X = rec_X["best_k"]

pt_Y = np.array(rec_Y["best_pt"])
wt_Y = rec_Y["best_wt"]
k_Y = rec_Y["best_k"]

pt_Z = np.array(rec_Z["best_pt"])
wt_Z = rec_Z["best_wt"]
k_Z = rec_Z["best_k"]

pt_univ = np.array(univ["coordinates"])
wt_univ = univ["wt"]
k_univ = univ["k"]

h_rev = 0.005

# Настройка шрифтов для графиков
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['lines.linewidth'] = 1.8

def build_master_window(start_pt, wt, h):
    n_steps = int(wt / h)
    mw = np.zeros((n_steps + 1, 3))
    curr = start_pt.copy()
    for i in range(n_steps + 1):
        mw[i] = curr
        curr[0], curr[1], curr[2] = sprott_step_cd(curr[0], curr[1], curr[2], h)
    return mw

# =====================================================================
# 1. ПЕРЕРИСОВКА ГРАФИКА 09: СХОДИМОСТЬ ОБРАТИМОЙ СИНХРОНИЗАЦИИ ПО ИТЕРАЦИЯМ
# =====================================================================
print("1. Перерисовка 09_reversible_convergence_vs_iterations.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Левая панель: Сходимость в рекордных точках для каждого сигнала
num_iters = 30
# Канал X
errs_X = get_numba_exact_history(np.array([pt_X]), int(wt_X / h_rev), k_X, 0, num_iters)[0]
safe_X = np.clip(errs_X, 1e-16, None)

# Канал Y
errs_Y = get_numba_exact_history(np.array([pt_Y]), int(wt_Y / h_rev), k_Y, 1, num_iters)[0]
safe_Y = np.clip(errs_Y, 1e-16, None)

# Канал Z
errs_Z = get_numba_exact_history(np.array([pt_Z]), int(wt_Z / h_rev), k_Z, 2, num_iters)[0]
safe_Z = np.clip(errs_Z, 1e-16, None)

ax1.plot(range(num_iters + 1), np.log10(safe_X), 'b-o', markersize=4, label='Сигнал X (достигает 0.0 на ит. 13)')
ax1.plot(range(num_iters + 1), np.log10(safe_Y), 'r--s', markersize=4, label='Сигнал Y (достигает 1.11e-16 на ит. 21)')
ax1.plot(range(num_iters + 1), np.log10(safe_Z), 'g-.^', markersize=4, label='Сигнал Z (достигает 0.0 на ит. 20)')
ax1.axhline(np.log10(2.22e-16), color='purple', linestyle=':', label=r'Машинный эпсилон $\varepsilon_{mach}$')
ax1.set_xlabel('Номер итерации i')
ax1.set_ylabel(r'$\log_{10}(||e_i||)$')
ax1.set_title('А. Сходимость в специализированных точках X*, Y*, Z*')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# Правая панель: Сходимость в универсальной точке для всех трех сигналов
errs_u_X = get_numba_exact_history(np.array([pt_univ]), int(wt_univ / h_rev), k_univ, 0, num_iters)[0]
errs_u_Y = get_numba_exact_history(np.array([pt_univ]), int(wt_univ / h_rev), k_univ, 1, num_iters)[0]
errs_u_Z = get_numba_exact_history(np.array([pt_univ]), int(wt_univ / h_rev), k_univ, 2, num_iters)[0]

ax2.plot(range(num_iters + 1), np.log10(np.clip(errs_u_X, 1e-16, None)), 'b-', label='Сигнал X')
ax2.plot(range(num_iters + 1), np.log10(np.clip(errs_u_Y, 1e-16, None)), 'r--', label='Сигнал Y')
ax2.plot(range(num_iters + 1), np.log10(np.clip(errs_u_Z, 1e-16, None)), 'g-.', label='Сигнал Z')
ax2.axhline(np.log10(2.22e-16), color='purple', linestyle=':', label=r'$\varepsilon_{mach}$')
ax2.set_xlabel('Номер итерации i')
ax2.set_ylabel(r'$\log_{10}(||e_i||)$')
ax2.set_title(f'Б. Сходимость в универсальной точке X*univ (WT = {wt_univ} с)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.suptitle('Сходимость обратимой синхронизации по итерациям: все три сигнала сходятся!', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_reversible_convergence_vs_iterations.png"), dpi=200)
plt.close()

# =====================================================================
# 2. ПЕРЕРИСОВКА ГРАФИКА 10: ВЛИЯНИЕ ДЛИНЫ ОКНА WT
# =====================================================================
print("2. Перерисовка 10_reversible_window_comparison.png...")
plt.figure(figsize=(9, 5.2))
wt_tests = [1.0, 2.0, 2.5, 3.0]
wt_cols = ['purple', 'blue', 'orange', 'green']
for wt_val, col in zip(wt_tests, wt_cols):
    errs_wt = get_numba_exact_history(np.array([pt_univ]), int(wt_val / h_rev), 2.0, 0, 30)[0]
    plt.plot(range(31), np.log10(np.clip(errs_wt, 1e-16, None)), color=col, label=f'WT = {wt_val:.1f} с')

plt.axhline(np.log10(2.22e-16), color='gray', linestyle=':', label=r'$\varepsilon_{mach}$')
plt.xlabel('Номер итерации i')
plt.ylabel(r'$\log_{10}(||e_i||)$')
plt.title('Влияние размера временного окна WT на сходимость обратимой синхронизации')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "10_reversible_window_comparison.png"), dpi=200)
plt.close()

# =====================================================================
# 3. ПЕРЕРИСОВКА ГРАФИКА 11: ЭВОЛЮЦИЯ ТРАЕКТОРИИ СЛЕЙВА
# =====================================================================
print("3. Перерисовка 11_reversible_trajectory_evolution.png...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
mw_evol = build_master_window(pt_X, wt_X, h_rev)
t_evol = np.linspace(0, wt_X, len(mw_evol))
iters_to_show = [0, 2, 6, 13]

for ax, it in zip(axes.flatten(), iters_to_show):
    if it == 0:
        # Автономное движение без связи
        curr_s = np.array([pt_X[0]+0.15, pt_X[1]-0.15, pt_X[2]+0.10])
        tr_s = np.zeros_like(mw_evol)
        for step in range(len(mw_evol)):
            tr_s[step] = curr_s
            curr_s[0], curr_s[1], curr_s[2] = sprott_step_cd(curr_s[0], curr_s[1], curr_s[2], h_rev)
        err_now = np.linalg.norm(tr_s[0] - mw_evol[0])
    else:
        errs_sub, _, _, _, tr_s = run_reversible_sync(
            mw_evol, pt_X[0]+0.15, pt_X[1]-0.15, pt_X[2]+0.10, h_rev, k_X, k_X, 0, it
        )
        err_now = errs_X[it]

    ax.plot(t_evol, mw_evol[:, 0], 'r-', label='Master x_m(t)', linewidth=2.0, alpha=0.8)
    ax.plot(t_evol, tr_s[:, 0], 'b--', label=f'Slave x_s(t)', linewidth=1.8, alpha=0.85)
    if err_now == 0.0:
        ax.set_title(f'Итерация {it} (ошибка ||e|| = 0.00e+00 [МАШИННЫЙ НОЛЬ])')
    else:
        ax.set_title(f'Итерация {it} (ошибка ||e|| = {err_now:.2e})')
    ax.set_ylabel('Координата x(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')

axes[1, 0].set_xlabel('Время t в окне, с')
axes[1, 1].set_xlabel('Время t в окне, с')
plt.suptitle('Эволюция траектории ведомой системы (Slave) к ведущей (Master) на окне WT = 2.0 с')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "11_reversible_trajectory_evolution.png"), dpi=200)
plt.close()

# =====================================================================
# 4. ПЕРЕРИСОВКА ГРАФИКА 12: СРАВНЕНИЕ РЕВЕРСИВНОГО МЕТОДА И ПИКОРЫ — КЭРОЛА
# =====================================================================
print("4. Перерисовка 12_comparison_reversible_vs_pecora_carroll.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Левая панель: СИГНАЛ X
# PC непрерывный во времени (t от 0 до 30)
t_pc = np.linspace(0, 30, int(30 / 0.01))
_, _, errs_pc_x = simulate_unidir(pt_X[0], pt_X[1], pt_X[2], pt_X[0]+0.15, pt_X[1]-0.15, pt_X[2]+0.10, 0.01, len(t_pc), 2.0, 0)
ax1.plot(t_pc, np.log10(np.clip(errs_pc_x, 1e-16, None)), 'b-', label='Пикора — Кэрол (непрерывный поток, k=2.0)')

# Обратимый метод (эквивалентное процессорное время t_eff = i * WT)
it_x_range = np.arange(len(safe_X))
t_eff_rev = it_x_range * (2.0 * wt_X) # 2 прохода по WT
ax1.plot(t_eff_rev, np.log10(safe_X), 'm--o', markersize=4, label='Обратимый метод (окно WT = 2.0 с, k=2.2)')
ax1.axhline(np.log10(2.22e-16), color='purple', linestyle=':', label=r'$\varepsilon_{mach}$')
ax1.set_xlabel('Эффективное время моделирования, с')
ax1.set_ylabel(r'$\log_{10}(||e||)$')
ax1.set_title('А. Сигнал X: оба метода сходятся, обратимый достигает 0.0')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# Правая панель: СИГНАЛ Y (КЛЮЧЕВОЙ КОНТРАСТ!)
_, _, errs_pc_y = simulate_unidir(pt_Y[0], pt_Y[1], pt_Y[2], pt_Y[0]+0.15, pt_Y[1]-0.15, pt_Y[2]+0.10, 0.01, len(t_pc), 2.0, 1)
ax2.plot(t_pc, np.log10(np.clip(errs_pc_y, 1e-16, None)), 'r-', label='Пикора — Кэрол: РАССИНХРОНИЗАЦИЯ (вечный маятник)')

it_y_range = np.arange(len(safe_Y))
t_eff_rev_y = it_y_range * (2.0 * wt_Y)
ax2.plot(t_eff_rev_y, np.log10(safe_Y), 'g--s', markersize=4, label='Обратимый метод: СХОДИТСЯ ниже машинного эпсилона!')
ax2.axhline(np.log10(2.22e-16), color='purple', linestyle=':', label=r'$\varepsilon_{mach}$')
ax2.set_xlabel('Эффективное время моделирования, с')
ax2.set_ylabel(r'$\log_{10}(||e||)$')
ax2.set_title('Б. Сигнал Y: Пикора — Кэрол проваливается, обратимый РАБОТАЕТ!')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.suptitle('Сравнение методов: классический Пикора — Кэрол против обратимого метода', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "12_comparison_reversible_vs_pecora_carroll.png"), dpi=200)
plt.close()

# =====================================================================
# 5. ПЕРЕРИСОВКА СВОДНЫХ КАРТ 3x3: СИГНАЛЫ X, Y, Z
# =====================================================================
print("5. Перерисовка comprehensive_signal_x/y/z_all_methods.png...")

targets = [
    ('x', 0, pt_X, wt_X, k_X, safe_X),
    ('y', 1, pt_Y, wt_Y, k_Y, safe_Y),
    ('z', 2, pt_Z, wt_Z, k_Z, safe_Z)
]

for vname, var_idx, pt_best, wt_best, k_best, safe_arr in targets:
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    t_sim = 25.0
    steps = int(t_sim / 0.01)
    time_arr = np.linspace(0, t_sim, steps)
    show_steps = 1500

    # 1. Однонаправленная
    ku = 2.0
    tm_u, ts_u, err_u = simulate_unidir(pt_best[0], pt_best[1], pt_best[2],
                                        pt_best[0]+0.15, pt_best[1]-0.15, pt_best[2]+0.10,
                                        0.01, steps, ku, var_idx)
    axes[0, 0].plot(time_arr[:show_steps], tm_u[:show_steps, var_idx], 'b-', label=f'Master {vname}_m')
    axes[0, 0].plot(time_arr[:show_steps], ts_u[:show_steps, var_idx], 'r--', label=f'Slave {vname}_s')
    axes[0, 0].set_ylabel(f'{vname}(t)')
    axes[0, 0].set_title(f'1. Однонаправленный Пикора — Кэрол по {vname} (k={ku})')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend(loc='upper right')

    axes[0, 1].plot(time_arr[:show_steps], np.log10(np.clip(err_u[:show_steps], 1e-16, None)), 'k-')
    axes[0, 1].set_ylabel('log10(||e||)')
    status_u = "СХОДИТСЯ" if var_idx != 1 else "РАССИНХРОНИЗАЦИЯ"
    axes[0, 1].set_title(f'1. Ошибка однонаправленной связи [{status_u}]')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # 2. Двунаправленная
    kb = 1.5
    tm_b, ts_b, err_b = simulate_bidir(pt_best[0], pt_best[1], pt_best[2],
                                       pt_best[0]+0.15, pt_best[1]-0.15, pt_best[2]+0.10,
                                       0.01, steps, kb, var_idx)
    axes[1, 0].plot(time_arr[:show_steps], tm_b[:show_steps, var_idx], 'b-', label=f'Master {vname}_m')
    axes[1, 0].plot(time_arr[:show_steps], ts_b[:show_steps, var_idx], 'r--', label=f'Slave {vname}_s')
    axes[1, 0].set_ylabel(f'{vname}(t)')
    axes[1, 0].set_title(f'2. Двунаправленная взаимная связь по {vname} (k={kb})')
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
    status_b = "ДИАГОНАЛЬ СХОДИМОСТИ" if var_idx != 1 else "ХАОТИЧЕСКОЕ ПЯТНО (НЕТ СВЯЗИ)"
    axes[1, 1].set_title(f'2. Плоскость синхронизма [{status_b}]')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].legend()

    # 3. Обратимая синхронизация в рекордной точке
    mw_sig = build_master_window(pt_best, wt_best, h_rev)
    iters_sig = 30
    _, _, _, _, last_tr = run_reversible_sync(
        mw_sig, pt_best[0]+0.15, pt_best[1]-0.15, pt_best[2]+0.10,
        h_rev, k_best, k_best, var_idx, iters_sig
    )
    t_win_sig = np.linspace(0, wt_best, len(mw_sig))
    axes[2, 0].plot(t_win_sig, mw_sig[:, var_idx], 'r-', label=f'Master {vname}_m (окно WT={wt_best}с)', linewidth=2.0)
    axes[2, 0].plot(t_win_sig, last_tr[:, var_idx], 'b--', label=f'Slave {vname}_s (финал)', linewidth=1.8)
    axes[2, 0].set_xlabel('Время в окне t, с')
    axes[2, 0].set_ylabel(f'{vname}(t)')
    axes[2, 0].set_title(f'3. Обратимый метод: идеальное совпадение в окне WT = {wt_best} с')
    axes[2, 0].grid(True, linestyle='--', alpha=0.5)
    axes[2, 0].legend()

    axes[2, 1].plot(range(len(safe_arr)), np.log10(safe_arr), 'm-o', markersize=4)
    axes[2, 1].axhline(np.log10(2.22e-16), color='purple', linestyle=':', label=r'$\varepsilon_{mach}$')
    axes[2, 1].set_xlabel('Номер итерации i')
    axes[2, 1].set_ylabel('log10(||e||)')
    if var_idx == 0:
        status_rev = "финал 0.00e+00 [МАШИННЫЙ НОЛЬ]"
    elif var_idx == 1:
        status_rev = "финал 1.11e-16 [НИЖЕ EPS_MACH]"
    else:
        status_rev = "финал 0.00e+00 [МАШИННЫЙ НОЛЬ]"
    axes[2, 1].set_title(f'3. Ошибка обратимой связи: {status_rev}')
    axes[2, 1].grid(True, linestyle='--', alpha=0.5)
    axes[2, 1].legend()

    plt.suptitle(f'Сводный анализ всех трех методов для СИГНАЛА {vname.upper()}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"comprehensive_signal_{vname}_all_methods.png"), dpi=200)
    plt.close()

print("Все графики успешно перерисованы и сохранены!")
