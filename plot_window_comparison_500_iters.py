#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Отдельный скрипт для построения динамики сходимости ошибки обратимой синхронизации
на длительном интервале в 500 итераций для различных длин временного окна WT.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
num_iters = 500
eps_mach = 2.220446049250313e-16
log_eps = np.log10(eps_mach)

pt_univ = [-1.286205, -1.101576, 0.025948]
k_univ = 2.0
windows = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
styles = ['-', '--', '-.', ':', '-', '--']
colors_wt = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

print("Моделирование динамики ошибки для различных окон WT на 500 итерациях...")

fig, ax = plt.subplots(figsize=(10, 6))

for wt, st, c in zip(windows, styles, colors_wt):
    wt_steps = int(round(wt / h))
    window = np.zeros((wt_steps + 1, 3))
    cx, cy, cz = pt_univ[0], pt_univ[1], pt_univ[2]
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
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], h, k_univ, 0)
        for step in range(wt_steps, 0, -1):
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, window[step, 0], window[step, 1], window[step, 2], -h, -k_univ, 0)
        errs[it] = np.sqrt((window[0, 0] - xs)**2 + (window[0, 1] - ys)**2 + (window[0, 2] - zs)**2)

    log_errs = np.log10(np.clip(errs, 1e-16, 1e2))
    ax.plot(range(num_iters + 1), log_errs, color=c, linestyle=st, lw=1.8, label=f'$W_T = {wt}$ с')
    print(f"-> WT = {wt:.1f} с: конечная ошибка на 500 итерации = 10^({log_errs[-1]:.2f})")

ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
ax.set_title('Зависимость сходимости ошибки от длины временного окна $W_T$ ($K = 2.0$, 500 итераций)', fontweight='bold', pad=10)
ax.set_xlabel('Номер итерации i', fontweight='bold')
ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
ax.set_xlim(-5, 505)
ax.set_ylim(-16.8, 3.0)
ax.set_yticks(np.arange(-16, 4, 2))
ax.legend(loc='lower left', framealpha=0.90)

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "10_reversible_window_comparison_500_iters.png")
plt.savefig(out_path, bbox_inches='tight')
plt.close()

print(f"\nГрафик успешно сохранен: {out_path}")
