"""
Скрипт проведения полного комплекса вычислительных экспериментов для ЛР №5:
- Исследование однонаправленной синхронизации Пикоры — Кэрола (Pecora-Carroll) по сигналам x, y, z;
- Исследование двунаправленной синхронизации по сигналам x, y, z;
- Исследование быстрой обратимой синхронизации (Time-Reversible Synchronization) по статье Butusov et al. (2025);
- Построение и сохранение всех графиков высокого разрешения.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Добавляем текущую директорию в путь
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sprott_sync import (
    sprott_step_cd,
    simulate_unidir,
    simulate_bidir,
    run_reversible_sync
)

PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Стиль оформления графиков
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 1.5


def run_all_experiments():
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №5: СИНХРОНИЗАЦИЯ СИСТЕМЫ СПРОТТА (CASE B)")
    print("Численный метод: Симметричный обратимый интегратор CD")
    print("=" * 70)

    h = 0.01  # Базовый шаг дискретизации
    t_transient = 100.0
    steps_transient = int(t_transient / h)

    # Установившееся состояние на аттракторе для ведущей системы
    xm, ym, zm = 0.2, 0.2, 0.2
    for _ in range(steps_transient):
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)

    x0_m, y0_m, z0_m = xm, ym, zm
    # Начальное смещение для ведомой системы
    x0_s = x0_m + 0.5
    y0_s = y0_m - 0.4
    z0_s = z0_m + 0.3

    print(f"Начальные условия Master: ({x0_m:.4f}, {y0_m:.4f}, {z0_m:.4f})")
    print(f"Начальные условия Slave:  ({x0_s:.4f}, {y0_s:.4f}, {z0_s:.4f})")
    e0_init = np.sqrt((x0_m - x0_s)**2 + (y0_m - y0_s)**2 + (z0_m - z0_s)**2)
    print(f"Начальная ошибка ||e(0)|| = {e0_init:.4f}\n")

    # =========================================================================
    # ЭКСПЕРИМЕНТ 1: Однонаправленная синхронизация Пикоры — Кэрола
    # =========================================================================
    print("-" * 70)
    print("ЭКСПЕРИМЕНТ 1: Однонаправленная синхронизация Пикоры — Кэрола (x, y, z)")
    print("-" * 70)

    t_sim = 60.0
    steps_sim = int(t_sim / h)
    time_arr = np.linspace(0, t_sim, steps_sim)

    k_values = np.linspace(0.0, 4.0, 81)
    mean_errors_unidir = {0: [], 1: [], 2: []}
    var_names = {0: 'x', 1: 'y', 2: 'z'}

    for var_idx in [0, 1, 2]:
        for k in k_values:
            _, _, errs = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k, var_idx)
            # Установившаяся ошибка по последним 30% времени
            tail = int(0.3 * steps_sim)
            mean_errors_unidir[var_idx].append(np.mean(errs[-tail:]))

    # График 1: Ошибка vs k для однонаправленной синхронизации
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, mean_errors_unidir[0], 'b-o', markersize=4, label='Связь по сигналу x', alpha=0.9)
    plt.plot(k_values, mean_errors_unidir[1], 'r--s', markersize=4, label='Связь по сигналу y', alpha=0.9)
    plt.plot(k_values, mean_errors_unidir[2], 'g-.^', markersize=4, label='Связь по сигналу z', alpha=0.9)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
    plt.xlabel('Коэффициент связи k')
    plt.ylabel('Средняя установившаяся ошибка ||e||')
    plt.title('Однонаправленная синхронизация Пикоры — Кэрола: Зависимость ошибки от k')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_unidirectional_error_vs_k.png"), dpi=200)
    plt.close()

    # Временные ряды для k=2.0 (выше порога)
    k_test = 2.0
    for var_idx in [0, 1, 2]:
        traj_m, traj_s, errs = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_test, var_idx)
        vname = var_names[var_idx]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(time_arr[:2000], traj_m[:2000, var_idx], 'b-', label=f'Master {vname}_m(t)', alpha=0.85)
        ax1.plot(time_arr[:2000], traj_s[:2000, var_idx], 'r--', label=f'Slave {vname}_s(t)', alpha=0.85)
        ax1.set_ylabel(f'Сигнал {vname}(t)')
        ax1.set_title(f'Однонаправленная связь по сигналу {vname} (k = {k_test}): Временные ряды')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        # Ошибка в лог-масштабе
        safe_errs = np.clip(errs, 1e-16, None)
        ax2.plot(time_arr[:2000], np.log10(safe_errs[:2000]), 'k-', alpha=0.85)
        ax2.set_xlabel('Время t, с')
        ax2.set_ylabel('log10(||e(t)||)')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"02_unidirectional_timeseries_{vname}.png"), dpi=200)
        plt.close()

    # Фазовые портреты диагонали синхронизации (u_s vs u_m)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, var_idx in enumerate([0, 1, 2]):
        vname = var_names[var_idx]
        traj_m, traj_s, _ = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_test, var_idx)
        # Отображаем после переходного процесса
        tail = steps_sim // 2
        axes[i].scatter(traj_m[tail:, var_idx], traj_s[tail:, var_idx], s=1, c='navy', alpha=0.6)
        lims = [min(traj_m[tail:, var_idx].min(), traj_s[tail:, var_idx].min()),
                max(traj_m[tail:, var_idx].max(), traj_s[tail:, var_idx].max())]
        axes[i].plot(lims, lims, 'r--', label=f'{vname}_s = {vname}_m')
        axes[i].set_xlabel(f'{vname}_m')
        axes[i].set_ylabel(f'{vname}_s')
        axes[i].set_title(f'Связь по сигналу {vname}')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].legend()

    plt.suptitle(f'Диагональ синхронизации при однонаправленной связи (k = {k_test})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_unidirectional_phase_portraits.png"), dpi=200)
    plt.close()
    print("-> Графики однонаправленной синхронизации сохранены.")

    # =========================================================================
    # ЭКСПЕРИМЕНТ 2: Двунаправленная (взаимная) синхронизация
    # =========================================================================
    print("-" * 70)
    print("ЭКСПЕРИМЕНТ 2: Двунаправленная синхронизация (x, y, z)")
    print("-" * 70)

    mean_errors_bidir = {0: [], 1: [], 2: []}
    for var_idx in [0, 1, 2]:
        for k in k_values:
            _, _, errs = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k, var_idx)
            tail = int(0.3 * steps_sim)
            mean_errors_bidir[var_idx].append(np.mean(errs[-tail:]))

    plt.figure(figsize=(9, 5))
    plt.plot(k_values, mean_errors_bidir[0], 'b-o', markersize=4, label='Взаимная связь по x', alpha=0.9)
    plt.plot(k_values, mean_errors_bidir[1], 'r--s', markersize=4, label='Взаимная связь по y', alpha=0.9)
    plt.plot(k_values, mean_errors_bidir[2], 'g-.^', markersize=4, label='Взаимная связь по z', alpha=0.9)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
    plt.xlabel('Коэффициент связи k')
    plt.ylabel('Средняя установившаяся ошибка ||e||')
    plt.title('Двунаправленная синхронизация: Зависимость ошибки от k')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_bidirectional_error_vs_k.png"), dpi=200)
    plt.close()

    # Сравнение сходимости x: однонаправленная vs двунаправленная
    traj_m_u, traj_s_u, errs_u = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, 0.5, 0)
    traj_m_b, traj_s_b, errs_b = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, 0.5, 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(time_arr[:1500], traj_m_b[:1500, 0], 'b-', label='Master x_m(t)', alpha=0.85)
    ax1.plot(time_arr[:1500], traj_s_b[:1500, 0], 'r--', label='Slave x_s(t)', alpha=0.85)
    ax1.set_ylabel('Сигнал x(t)')
    ax1.set_title('Двунаправленная связь по сигналу x (k = 0.5): Временные ряды')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    ax2.plot(time_arr[:1500], np.log10(np.clip(errs_u[:1500], 1e-16, None)), 'g--', label='Однонаправленная (k=0.5)', alpha=0.85)
    ax2.plot(time_arr[:1500], np.log10(np.clip(errs_b[:1500], 1e-16, None)), 'm-', label='Двунаправленная (k=0.5)', alpha=0.85)
    ax2.set_xlabel('Время t, с')
    ax2.set_ylabel('log10(||e(t)||)')
    ax2.set_title('Сравнение скорости сходимости ошибки')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "07_bidirectional_timeseries_x.png"), dpi=200)
    plt.close()

    # Фазовые портреты двунаправленной синхронизации
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, var_idx in enumerate([0, 1, 2]):
        vname = var_names[var_idx]
        traj_m, traj_s, _ = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_test, var_idx)
        tail = steps_sim // 2
        axes[i].scatter(traj_m[tail:, var_idx], traj_s[tail:, var_idx], s=1, c='darkred', alpha=0.6)
        lims = [min(traj_m[tail:, var_idx].min(), traj_s[tail:, var_idx].min()),
                max(traj_m[tail:, var_idx].max(), traj_s[tail:, var_idx].max())]
        axes[i].plot(lims, lims, 'k--', label=f'{vname}_s = {vname}_m')
        axes[i].set_xlabel(f'{vname}_m')
        axes[i].set_ylabel(f'{vname}_s')
        axes[i].set_title(f'Взаимная связь по {vname}')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].legend()

    plt.suptitle(f'Диагональ синхронизации при двунаправленной связи (k = {k_test})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_bidirectional_phase_portraits.png"), dpi=200)
    plt.close()
    print("-> Графики двунаправленной синхронизации сохранены.")

    # =========================================================================
    # ЭКСПЕРИМЕНТ 3: Быстрая обратимая синхронизация (Time-Reversible Sync)
    # =========================================================================
    print("-" * 70)
    print("ЭКСПЕРИМЕНТ 3: Быстрая обратимая синхронизация (Butusov et al. 2025)")
    print("-" * 70)

    # Запись окна данных ведущей системы длительностью WT
    WT = 1.0  # секунды
    h_rev = 0.005  # шаг для обратимой схемы
    N_window = int(WT / h_rev)
    master_window = np.zeros((N_window + 1, 3))

    xm, ym, zm = x0_m, y0_m, z0_m
    for i in range(N_window + 1):
        master_window[i] = [xm, ym, zm]
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h_rev)

    # Начальное отклонение ведомой системы при t=0
    x0_s_rev = master_window[0, 0] + 0.15
    y0_s_rev = master_window[0, 1] - 0.15
    z0_s_rev = master_window[0, 2] + 0.15
    num_iters = 80

    # 1. Сходимость по итерациям для каждого из сигналов x, y, z
    plt.figure(figsize=(9, 5.5))
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    styles = {0: '-', 1: '--', 2: '-.'}

    for var_idx in [0, 1, 2]:
        vname = var_names[var_idx]
        K_val = 1.5
        iter_errs, _, _, _, _ = run_reversible_sync(
            master_window, x0_s_rev, y0_s_rev, z0_s_rev,
            h_rev, K_val, K_val, var_idx, num_iters
        )
        delta_log_err = np.log10(np.clip(iter_errs, 1e-16, None)) - np.log10(iter_errs[0])
        plt.plot(range(num_iters + 1), delta_log_err, color=colors[var_idx],
                 linestyle=styles[var_idx], label=f'Сигнал {vname} (K = {K_val})')

    plt.xlabel('Номер итерации прямого-обратного прохода i')
    plt.ylabel(r'$\Delta \log_{10}(E) = \log_{10}(e_i) - \log_{10}(e_0)$')
    plt.title(f'Обратимая синхронизация: Сходимость по итерациям (WT = {WT} с, h = {h_rev})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "09_reversible_convergence_vs_iterations.png"), dpi=200)
    plt.close()

    # 2. Исследование влияния длины окна WT (0.2, 0.5, 1.0, 2.0 с)
    plt.figure(figsize=(9, 5.5))
    wt_list = [0.2, 0.5, 1.0, 2.0]
    wt_colors = ['purple', 'blue', 'orange', 'green']

    for wt, col in zip(wt_list, wt_colors):
        n_w = int(wt / h_rev)
        mw = np.zeros((n_w + 1, 3))
        xm_t, ym_t, zm_t = x0_m, y0_m, z0_m
        for i in range(n_w + 1):
            mw[i] = [xm_t, ym_t, zm_t]
            xm_t, ym_t, zm_t = sprott_step_cd(xm_t, ym_t, zm_t, h_rev)

        iter_errs, _, _, _, _ = run_reversible_sync(
            mw, x0_s_rev, y0_s_rev, z0_s_rev,
            h_rev, 1.5, 1.5, 2, 60  # связь по z
        )
        delta_log = np.log10(np.clip(iter_errs, 1e-16, None)) - np.log10(iter_errs[0])
        plt.plot(range(61), delta_log, color=col, label=f'WT = {wt:.1f} с')

    plt.xlabel('Номер итерации i')
    plt.ylabel(r'$\Delta \log_{10}(E)$')
    plt.title('Влияние размера временного окна WT на сходимость (связь по z)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "10_reversible_window_comparison.png"), dpi=200)
    plt.close()

    # 3. Эволюция траектории слейва на окне WT при росте итераций
    # Подобно Рис. 1 и Рис. 6 статьи Butusov et al. (2025)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    t_win = np.linspace(0, WT, N_window + 1)
    test_iters_list = [0, 5, 20, 60]

    for ax, target_it in zip(axes.flatten(), test_iters_list):
        if target_it == 0:
            # Автономное движение слейва от начальной точки без синхронизации
            xs_curr, ys_curr, zs_curr = x0_s_rev, y0_s_rev, z0_s_rev
            slave_traj_0 = np.zeros((N_window + 1, 3))
            for i in range(N_window + 1):
                slave_traj_0[i] = [xs_curr, ys_curr, zs_curr]
                xs_curr, ys_curr, zs_curr = sprott_step_cd(xs_curr, ys_curr, zs_curr, h_rev)
            traj_to_plot = slave_traj_0
            curr_err = np.linalg.norm([x0_s_rev, y0_s_rev, z0_s_rev] - master_window[0])
        else:
            _, _, _, _, traj_to_plot = run_reversible_sync(
                master_window, x0_s_rev, y0_s_rev, z0_s_rev,
                h_rev, 1.5, 1.5, 2, target_it
            )
            curr_err = np.linalg.norm(traj_to_plot[0] - master_window[0])

        ax.plot(t_win, master_window[:, 2], 'b-', label='Master z_m(t)', linewidth=2.0, alpha=0.9)
        ax.plot(t_win, traj_to_plot[:, 2], 'r--', label='Slave z_s(t)', linewidth=1.8, alpha=0.85)
        ax.set_title(f'Итерация {target_it} (Ошибка при t=0: {curr_err:.3e})')
        ax.set_xlabel('Время в окне t, с')
        ax.set_ylabel('Сигнал z')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right')

    plt.suptitle('Эволюция траектории ведомой системы на окне WT при обратимой синхронизации')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "11_reversible_trajectory_evolution.png"), dpi=200)
    plt.close()

    # 4. Сравнение с обычной синхронизацией Пикоры-Кэрола на коротком интервале времени
    # Подобно Рис. 5 статьи Butusov et al. (2025)
    # Обычный Pecora-Carroll бежит вперед по времени, а Time-Reversible многократно проходит окно WT
    total_time_pc = 30.0
    steps_pc = int(total_time_pc / h_rev)
    time_pc = np.linspace(0, total_time_pc, steps_pc)
    _, _, errs_pc = simulate_unidir(
        master_window[0, 0], master_window[0, 1], master_window[0, 2],
        x0_s_rev, y0_s_rev, z0_s_rev,
        h_rev, steps_pc, 1.5, 2
    )

    # Запускаем обратимую синхронизацию на num_iters итераций
    iter_errs_comp, _, _, _, _ = run_reversible_sync(
        master_window, x0_s_rev, y0_s_rev, z0_s_rev,
        h_rev, 1.5, 1.5, 2, num_iters
    )

    plt.figure(figsize=(9, 5))
    plt.plot(time_pc, np.log10(np.clip(errs_pc, 1e-16, None)), 'b-', label='Обычный Pecora-Carroll (по времени t)')
    # Для обратимой синхронизации сопоставляем эквивалентный вычислительный бюджет:
    # 1 итерация = 2 прохода по WT = эквивалент 2 * WT времени
    equiv_time_rev = np.array(range(num_iters + 1)) * (2.0 * WT)
    plt.plot(equiv_time_rev, np.log10(np.clip(iter_errs_comp, 1e-16, None)), 'r--o', markersize=4,
             label=f'Обратимая синхронизация (на окне WT = {WT} с)')
    plt.axvline(WT, color='green', linestyle=':', label=f'Длина окна данных WT = {WT} с')
    plt.xlabel('Эквивалентное вычислительное время (интеграция)')
    plt.ylabel('log10(||e||)')
    plt.title('Сравнение обратимой синхронизации с классическим методом Пикоры — Кэрола')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "12_comparison_reversible_vs_pecora_carroll.png"), dpi=200)
    plt.close()
    print("-> Графики обратимой синхронизации сохранены.")

    print("\n" + "=" * 70)
    print("ВСЕ ВЫЧИСЛИТЕЛЬНЫЕ ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ!")
    print(f"Все 12 графиков сохранены в папку: {PLOTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    run_all_experiments()
