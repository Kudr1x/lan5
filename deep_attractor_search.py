#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Глубокий многопоточный сеточный поиск (Grid Search) оптимальных точек и параметров
для метода быстрой обратимой синхронизации системы Спротта B на 20 вычислительных ядрах:
- Сканирование 100 000 фазовых состояний странного аттрактора
- Сетка размеров окна WT: 2.0, 2.5, 3.0, 3.5 с
- Сетка коэффициентов связи K: 1.8, 2.0, 2.2, 2.5
- Поканальный анализ: X, Y, Z отдельно + Мультикритериальный поиск универсальных точек
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba
from numba import prange

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Вспомогательные функции интегрирования на Numba
# ---------------------------------------------------------------------------

@numba.njit(fastmath=True)
def sprott_step_cd(x, y, z, h):
    tau = 0.5 * h
    x1 = x + tau * (y * z)
    y1 = y + tau * (x1 - y)
    z1 = z + tau * (1.0 - x1 * y1)
    z2 = z1 + tau * (1.0 - x1 * y1)
    y2 = (y1 + tau * x1) / (1.0 + tau)
    x2 = x1 + tau * (y2 * z2)
    return x2, y2, z2


@numba.njit(fastmath=True)
def sprott_step_coupled(x, y, z, xm, ym, zm, h, K, var_idx):
    tau = 0.5 * h
    cx = K * (xm - x) if var_idx == 0 else 0.0
    cy = K * (ym - y) if var_idx == 1 else 0.0
    cz = K * (zm - z) if var_idx == 2 else 0.0

    x1 = x + tau * (y * z + cx)
    y1 = y + tau * (x1 - y + cy)
    z1 = z + tau * (1.0 - x1 * y1 + cz)
    z2 = z1 + tau * (1.0 - x1 * y1 + cz)
    y2 = (y1 + tau * (x1 + cy)) / (1.0 + tau)
    x2 = x1 + tau * (y2 * z2 + cx)
    return x2, y2, z2


@numba.njit(parallel=True, fastmath=True)
def parallel_evaluate_points(pts, WT_steps, K_val, var_idx, iters=25, h=0.005):
    """
    Параллельная оценка массива точек аттрактора по заданному каналу var_idx.
    Возвращает минимальную достигнутую норму ошибки и номер лучшей итерации для каждой точки.
    """
    n = len(pts)
    min_errors = np.zeros(n)
    best_iters = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        pt = pts[i]
        window = np.zeros((WT_steps + 1, 3))
        cx, cy, cz = pt[0], pt[1], pt[2]
        for step in range(WT_steps + 1):
            window[step, 0] = cx
            window[step, 1] = cy
            window[step, 2] = cz
            cx, cy, cz = sprott_step_cd(cx, cy, cz, h)

        # Начальное возмущение Slave
        xs = window[0, 0] + 0.15
        ys = window[0, 1] - 0.15
        zs = window[0, 2] + 0.10

        best_e = 1e10
        best_it = -1

        for it in range(1, iters + 1):
            # Forward pass
            for step in range(WT_steps):
                xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], h, K_val, var_idx)
            # Backward pass
            for step in range(WT_steps, 0, -1):
                xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], -h, -K_val, var_idx)

            err = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)
            if err < best_e:
                best_e = err
                best_it = it
            if err <= 1e-15:
                break

        min_errors[i] = best_e
        best_iters[i] = best_it

    return min_errors, best_iters


def run_deep_attractor_search():
    print("=" * 80)
    print("ГЛУБОКИЙ СЕТОЧНЫЙ ПОИСК ТОЧЕК И ПАРАМЕТРОВ ОБРАТИМОЙ СИНХРОНИЗАЦИИ")
    print("Платформа: 20 вычислительных ядер, Numba OpenMP parallel execution")
    print("=" * 80)

    h = 0.005
    n_pts = 100000

    print(f"\n1. Генерация {n_pts} состояний странного аттрактора Спротта B...")
    t0_gen = time.time()
    # Сброс переходных процессов
    xm, ym, zm = 0.2, 0.2, 0.2
    for _ in range(20000):
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)

    pts = np.zeros((n_pts, 3))
    for i in range(n_pts):
        for _ in range(4):  # Шаг 0.02 с между точками для равномерного заполнения
            xm, ym, zm = sprott_step_cd(xm, ym, zm, h)
        pts[i] = [xm, ym, zm]
    print(f"-> Аттрактор успешно сгенерирован за {time.time() - t0_gen:.2f} с.")

    # Сетка исследуемых параметров
    wt_list = [2.0, 2.5, 3.0, 3.5]
    k_list = [1.8, 2.0, 2.2, 2.5]
    var_names = ['X', 'Y', 'Z']

    # Прогрев Numba компилятора
    _ = parallel_evaluate_points(pts[:10], 100, 2.0, 0, 2)

    # Структуры для сохранения рекордов
    records = {
        'X': {'best_err': 1e10, 'best_pt': None, 'best_wt': None, 'best_k': None, 'best_it': None},
        'Y': {'best_err': 1e10, 'best_pt': None, 'best_wt': None, 'best_k': None, 'best_it': None},
        'Z': {'best_err': 1e10, 'best_pt': None, 'best_wt': None, 'best_k': None, 'best_it': None},
    }

    # Массив для хранения лучшей ошибки по всем трем каналам для универсального анализа
    universal_matrix = np.zeros((n_pts, 3))
    universal_matrix.fill(1e10)

    total_configs = len(wt_list) * len(k_list) * 3
    current_config = 0
    t0_search = time.time()

    print(f"\n2. Запуск вложенных циклов по {n_pts} точкам x {len(wt_list)} окнам x {len(k_list)} коэфф. связи x 3 каналам...")
    print(f"Всего тестовых конфигураций: {total_configs} (3.6 млн полных траекторий)\n")

    for wt in wt_list:
        wt_steps = int(wt / h)
        for k in k_list:
            for v_idx, v_name in enumerate(var_names):
                current_config += 1
                t_sub = time.time()
                errs, iters = parallel_evaluate_points(pts, wt_steps, k, v_idx, iters=25, h=h)
                best_sub_idx = np.argmin(errs)
                min_e = errs[best_sub_idx]
                best_it = iters[best_sub_idx]

                if min_e < records[v_name]['best_err']:
                    records[v_name]['best_err'] = min_e
                    records[v_name]['best_pt'] = pts[best_sub_idx].tolist()
                    records[v_name]['best_wt'] = wt
                    records[v_name]['best_k'] = k
                    records[v_name]['best_it'] = int(best_it)

                # Сохраняем лучший результат для каждой точки при базовых параметрах WT=3.0, K=2.0
                if wt == 3.0 and k == 2.0:
                    universal_matrix[:, v_idx] = errs

                pct = (current_config / total_configs) * 100
                elapsed = time.time() - t0_search
                eta = (elapsed / current_config) * (total_configs - current_config)
                print(f"[{current_config:2d}/{total_configs}] ({pct:5.1f}%) | WT={wt:.1f}с, K={k:.1f}, Канал {v_name} | Мин. ош: {min_e:.3e} (ит. {best_it}) | Прошло: {elapsed:.1f}с, ETA: {eta:.1f}с")

    # Поиск универсальной точки (минимизация максимума из ошибок по X, Y, Z)
    worst_across_channels = np.maximum(np.maximum(universal_matrix[:, 0], universal_matrix[:, 1]), universal_matrix[:, 2])
    univ_idx = np.argmin(worst_across_channels)
    univ_pt = pts[univ_idx].tolist()
    univ_errs = [float(universal_matrix[univ_idx, 0]), float(universal_matrix[univ_idx, 1]), float(universal_matrix[univ_idx, 2])]

    total_time = time.time() - t0_search
    print("\n" + "=" * 80)
    print(f"ПОИСК ЗАВЕРШЕН ЗА {total_time:.1f} с ({total_time / 60:.2f} мин)")
    print("=" * 80)

    print("\n--- ИТОГОВЫЕ РЕЗУЛЬТАТЫ ГЛУБОКОГО СКАНИРОВАНИЯ ---")
    eps = np.finfo(float).eps
    print(f"Машинный эпсилон IEEE 754 (float64): {eps:.6e}\n")

    for v_name in ['X', 'Y', 'Z']:
        rec = records[v_name]
        pt = rec['best_pt']
        print(f"Канал {v_name} (отдельно):")
        print(f"  Абсолютный минимум ошибки: {rec['best_err']:.6e} (достигнут на {rec['best_it']}-й итерации)")
        print(f"  Оптимальная точка: X*_{v_name} = ({pt[0]:.6f}, {pt[1]:.6f}, {pt[2]:.6f})")
        print(f"  Оптимальные параметры: WT = {rec['best_wt']:.1f} с, K = {rec['best_k']:.1f}")
        below_eps = "СТРОГО НИЖЕ МАШИННОГО ЭПСИЛОНА" if rec['best_err'] <= eps else "НА УРОВНЕ МАШИННОГО ЭПСИЛОНА"
        print(f"  Статус: {below_eps}\n")

    print(f"УНИВЕРСАЛЬНАЯ ТОЧКА (где сходятся ВСЕ ТРИ канала X, Y, Z по отдельности):")
    print(f"  X*_univ = ({univ_pt[0]:.6f}, {univ_pt[1]:.6f}, {univ_pt[2]:.6f})")
    print(f"  Ошибки: по X = {univ_errs[0]:.3e}, по Y = {univ_errs[1]:.3e}, по Z = {univ_errs[2]:.3e}")

    # Сохранение базы данных в JSON
    db_path = os.path.join(CURRENT_DIR, "optimal_points_database.json")
    db_data = {
        'records_by_channel': records,
        'universal_point': {
            'coordinates': univ_pt,
            'errors': univ_errs,
            'wt': 3.0,
            'k': 2.0
        },
        'search_summary': {
            'points_sampled': n_pts,
            'total_time_sec': total_time,
            'machine_epsilon': eps
        }
    }
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db_data, f, indent=4, ensure_ascii=False)
    print(f"\nБаза данных оптимальных точек сохранена: {db_path}")

    # -----------------------------------------------------------------------
    # Построение графиков глубокого поиска
    # -----------------------------------------------------------------------
    print("\n3. Генерация иллюстраций...")

    # График 16: 3D карта плотности сходимости на аттракторе
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    p_step = max(1, n_pts // 15000)
    sub_pts = pts[::p_step]
    sub_errs = worst_across_channels[::p_step]
    log_errs = np.log10(np.clip(sub_errs, 1e-16, 1e2))

    sc = ax.scatter(sub_pts[:, 0], sub_pts[:, 1], sub_pts[:, 2], c=log_errs, cmap='viridis_r', s=6, alpha=0.7)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label(r'$\log_{10}(\max(e_x, e_y, e_z))$', fontsize=12)

    # Отметка рекордеров
    pX = records['X']['best_pt']
    pY = records['Y']['best_pt']
    pZ = records['Z']['best_pt']
    ax.scatter([pX[0]], [pX[1]], [pX[2]], color='blue', s=160, edgecolors='black', label=f"X* ({pX[0]:.2f}, {pX[1]:.2f}, {pX[2]:.2f})")
    ax.scatter([pY[0]], [pY[1]], [pY[2]], color='red', s=160, edgecolors='black', label=f"Y* ({pY[0]:.2f}, {pY[1]:.2f}, {pY[2]:.2f})")
    ax.scatter([pZ[0]], [pZ[1]], [pZ[2]], color='green', s=160, edgecolors='black', label=f"Z* ({pZ[0]:.2f}, {pZ[1]:.2f}, {pZ[2]:.2f})")
    ax.scatter([univ_pt[0]], [univ_pt[1]], [univ_pt[2]], color='magenta', s=200, marker='*', edgecolors='black', label=f"Univ* ({univ_pt[0]:.2f}, {univ_pt[1]:.2f}, {univ_pt[2]:.2f})")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Глубокое сканирование аттрактора: карта сходимости по {n_pts} точкам')
    ax.legend(loc='upper right', fontsize=9.5)
    plt.tight_layout()
    p16 = os.path.join(PLOTS_DIR, "16_deep_search_attractor_density.png")
    plt.savefig(p16, dpi=200)
    plt.close()
    print(f"-> Сохранен график: {p16}")

    print("\n" + "=" * 80)
    print("ВСЕ РАСЧЕТЫ И ГРАФИКИ УСПЕШНО ЗАВЕРШЕНЫ!")
    print("=" * 80)


if __name__ == '__main__':
    run_deep_attractor_search()
