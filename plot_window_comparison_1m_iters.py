#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт исследования сходимости обратимой синхронизации на 1 000 000 (1M) итераций
с логарифмической шкалой по оси абсцисс (число итераций i от 1 до 10^6).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit

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
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.6,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

C_PURPLE = '#8e44ad'
h = 0.005
num_iters = 1_000_000
eps_mach = 2.220446049250313e-16
log_eps = np.log10(eps_mach)

pt_univ = np.array([-1.286205, -1.101576, 0.025948])
k_univ = 2.0
windows = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
styles = ['-', '--', '-.', ':', '-', '--']
colors_wt = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

# Логарифмическая сетка точек для построения графика:
# Берем все целые числа от 1 до 500, а затем 4500 логарифмически распределенных точек до 10^6
log_sample_indices = np.unique(np.concatenate([
    np.arange(1, 501, dtype=np.int64),
    np.geomspace(500, num_iters, 4500).astype(np.int64)
]))
sample_mask = np.zeros(num_iters + 1, dtype=np.bool_)
sample_mask[log_sample_indices] = True
sample_mask[0] = True

@njit(fastmath=True)
def run_simulation_sampled(window, wt_steps, num_iters, k, var_idx, h, sample_indices):
    xs = window[0, 0] + 0.15
    ys = window[0, 1] - 0.15
    zs = window[0, 2] + 0.10
    
    n_samples = len(sample_indices)
    sampled_errs = np.zeros(n_samples)
    
    e0 = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)
    sample_ptr = 0
    if sample_indices[0] == 0:
        sampled_errs[0] = e0
        sample_ptr = 1

    next_sample_it = sample_indices[sample_ptr]
    consecutive_zeros = 0
    
    for it in range(1, num_iters + 1):
        for step in range(wt_steps):
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], h, k, var_idx)
        for step in range(wt_steps, 0, -1):
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], -h, -k, var_idx)
        
        e = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)
        
        if e < 1e-15:
            consecutive_zeros += 1
        else:
            consecutive_zeros = 0
            
        if it == next_sample_it:
            sampled_errs[sample_ptr] = e
            sample_ptr += 1
            if sample_ptr < n_samples:
                next_sample_it = sample_indices[sample_ptr]
                
        # Если достигли машинного предела и зафиксировались более чем на 300 итераций
        if consecutive_zeros > 300:
            while sample_ptr < n_samples:
                sampled_errs[sample_ptr] = e
                sample_ptr += 1
            break
            
    return sampled_errs

# Warmup JIT
_dummy_win = np.zeros((10, 3))
_dummy_idx = np.array([0, 1, 2], dtype=np.int64)
run_simulation_sampled(_dummy_win, 9, 2, 2.0, 0, h, _dummy_idx)

print("Моделирование динамики ошибки на 1 000 000 итераций...")

fig, ax = plt.subplots(figsize=(11, 6.5))

# Подготовка массива x для логарифмической шкалы
plot_x = log_sample_indices.copy()
# Начинаем с итерации 1 для строгого логарифмического масштаба
mask_pos = plot_x >= 1
plot_x_pos = plot_x[mask_pos]

for wt, st, c in zip(windows, styles, colors_wt):
    wt_steps = int(round(wt / h))
    window = np.zeros((wt_steps + 1, 3))
    cx, cy, cz = pt_univ[0], pt_univ[1], pt_univ[2]
    for step in range(wt_steps + 1):
        window[step] = [cx, cy, cz]
        cx, cy, cz = sprott_step_cd(cx, cy, cz, h)

    sampled_errs = run_simulation_sampled(window, wt_steps, num_iters, k_univ, 0, h, log_sample_indices)
    sampled_errs_pos = sampled_errs[mask_pos]
    
    log_errs = np.log10(np.clip(sampled_errs_pos, 1e-16, 1e2))
    ax.plot(plot_x_pos, log_errs, color=c, linestyle=st, lw=1.8, label=f'$W_T = {wt}$ с')
    print(f"-> WT = {wt:.1f} с: ошибка на 10^6 итераций = 10^({log_errs[-1]:.2f})")

ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')

# Заголовок строго БЕЗ указания числа итераций
ax.set_title('Зависимость сходимости ошибки от длины временного окна $W_T$ ($K = 2.0$)', fontweight='bold', pad=10)
ax.set_xlabel('Номер итерации i', fontweight='bold')
ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')

# Логарифмический масштаб по оси абсцисс
ax.set_xscale('log')
ax.set_xlim(1, 1_000_000)

ax.set_ylim(-16.8, 3.0)
ax.set_yticks(np.arange(-16, 4, 2))

# Легенда снизу под графиком
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=4, framealpha=0.95, fontsize=10)

plt.subplots_adjust(top=0.92, bottom=0.22, left=0.09, right=0.97)

out_path = os.path.join(PLOTS_DIR, "10_reversible_window_comparison_1m_iters.png")
plt.savefig(out_path, bbox_inches='tight')
plt.close()

print(f"\nГрафик успешно сохранен: {out_path}")
