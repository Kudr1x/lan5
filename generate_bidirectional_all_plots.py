#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Генерация всех трех графиков двунаправленной (взаимной) синхронизации
в едином каноническом стиле:
1. 07_bidirectional_timeseries_x.png (канал X)
2. 07_bidirectional_timeseries_y.png (канал Y)
3. 07_bidirectional_timeseries_z.png (канал Z)
4. 07_bidirectional_timeseries_all_3.png (сводный 3-панельный вариант)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = '/home/bobbert/Документы/leti/mnds/lab4/lab5'
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)

from sprott_sync import sprott_step_cd, simulate_unidir, simulate_bidir

# Параметры интегрирования
h = 0.01
t_transient = 100.0
steps_transient = int(t_transient / h)

# Выход на аттрактор ведущей системы
xm, ym, zm = 0.2, 0.2, 0.2
for _ in range(steps_transient):
    xm, ym, zm = sprott_step_cd(xm, ym, zm, h)

x0_m, y0_m, z0_m = xm, ym, zm
x0_s = x0_m + 0.5
y0_s = y0_m - 0.4
z0_s = z0_m + 0.3

t_sim = 60.0
steps_sim = int(t_sim / h)
time_arr = np.linspace(0, t_sim, steps_sim)
k_val = 0.5

# Окно отображения: 15 секунд (1500 шагов при h=0.01)
n_pts = 1500
t_window = time_arr[:n_pts]

var_names = ['x', 'y', 'z']

# Общие настройки Matplotlib для строго единого стиля
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.6
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("Генерация трех графиков в едином стиле...")

for var_idx, vname in enumerate(var_names):
    traj_m_u, traj_s_u, errs_u = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_val, var_idx)
    traj_m_b, traj_s_b, errs_b = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_val, var_idx)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Верхний график: Временные ряды Master и Slave
    ax1.plot(t_window, traj_m_b[:n_pts, var_idx], 'b-', label=f'Master {vname}_m(t)', alpha=0.9, lw=1.8)
    ax1.plot(t_window, traj_s_b[:n_pts, var_idx], 'r--', label=f'Slave {vname}_s(t)', alpha=0.9, lw=1.8)
    ax1.set_ylabel(f'Сигнал {vname}(t)', fontsize=12)
    ax1.set_title(f'Двунаправленная связь по сигналу {vname} (k = {k_val}): Временные ряды', fontsize=13, pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Нижний график: Сравнение скорости сходимости ошибки
    safe_u = np.clip(errs_u[:n_pts], 1e-16, None)
    safe_b = np.clip(errs_b[:n_pts], 1e-16, None)

    ax2.plot(t_window, np.log10(safe_u), 'g--', label=f'Однонаправленная (k={k_val})', alpha=0.9, lw=1.8)
    ax2.plot(t_window, np.log10(safe_b), 'm-', label=f'Двунаправленная (k={k_val})', alpha=0.9, lw=2.0)
    ax2.set_xlabel('Время t, с', fontsize=12)
    ax2.set_ylabel('log10(||e(t)||)', fontsize=12)
    ax2.set_title('Сравнение скорости сходимости ошибки', fontsize=13, pad=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Размещение легенды в зависимости от траектории
    if var_idx == 1:
        # Для Y ошибки не падают
        ax2.legend(loc='upper right', framealpha=0.9)
    else:
        ax2.legend(loc='lower left', framealpha=0.9)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, f"07_bidirectional_timeseries_{vname}.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"-> Сохранен график: {out_file}")

# Сводный 3-панельный график для всех трех сигналов
fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex='col')

for var_idx, vname in enumerate(var_names):
    traj_m_u, traj_s_u, errs_u = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_val, var_idx)
    traj_m_b, traj_s_b, errs_b = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_val, var_idx)

    # Колонка 1: Временные ряды
    ax1 = axes[var_idx, 0]
    ax1.plot(t_window, traj_m_b[:n_pts, var_idx], 'b-', label=f'Master {vname}_m(t)', lw=1.8)
    ax1.plot(t_window, traj_s_b[:n_pts, var_idx], 'r--', label=f'Slave {vname}_s(t)', lw=1.8)
    ax1.set_ylabel(f'Сигнал {vname}(t)', fontsize=11)
    ax1.set_title(f'Канал {vname.upper()} (k = {k_val}): Временные ряды', fontsize=12, pad=6)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9)

    # Колонка 2: Ошибки
    ax2 = axes[var_idx, 1]
    safe_u = np.clip(errs_u[:n_pts], 1e-16, None)
    safe_b = np.clip(errs_b[:n_pts], 1e-16, None)
    ax2.plot(t_window, np.log10(safe_u), 'g--', label=f'Однонаправленная (k={k_val})', lw=1.8)
    ax2.plot(t_window, np.log10(safe_b), 'm-', label=f'Двунаправленная (k={k_val})', lw=2.0)
    ax2.set_ylabel('log10(||e(t)||)', fontsize=11)
    ax2.set_title(f'Канал {vname.upper()}: Сравнение затухания ошибки', fontsize=12, pad=6)
    ax2.grid(True, linestyle='--', alpha=0.5)
    if var_idx == 1:
        ax2.legend(loc='upper right', fontsize=9)
    else:
        ax2.legend(loc='lower left', fontsize=9)

axes[2, 0].set_xlabel('Время t, с', fontsize=12)
axes[2, 1].set_xlabel('Время t, с', fontsize=12)

plt.suptitle('Двунаправленная (взаимная) синхронизация системы Спротта B по всем трем каналам (k = 0.5)', fontsize=14, y=0.995)
plt.tight_layout()
summary_file = os.path.join(PLOTS_DIR, "07_bidirectional_timeseries_all_3.png")
plt.savefig(summary_file, bbox_inches='tight')
plt.close()
print(f"-> Сохранен сводный 3-канальный график: {summary_file}")
