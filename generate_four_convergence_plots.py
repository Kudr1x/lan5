#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт генерации 4 абсолютно чистых графиков сходимости ошибки по поколениям (итерациям) в лог-масштабе
БЕЗ любых текстовых выносок, стрелок, плашек и примечаний:
1. plots/09_convergence_generation_x.png
2. plots/09_convergence_generation_y.png
3. plots/09_convergence_generation_z.png
4. plots/09_convergence_generation_universal.png
а также двухпанельного plots/09_reversible_convergence_vs_iterations.png
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE_DIR = '/home/bobbert/Документы/leti/mnds/lab4/lab5'
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
sys.path.insert(0, BASE_DIR)

from sprott_sync import sprott_step_cd, sprott_step_coupled

# База оптимальных точек
DB_PATH = os.path.join(BASE_DIR, 'optimal_points_database.json')
with open(DB_PATH, 'r', encoding='utf-8') as f:
    db = json.load(f)

h = 0.005
num_iters = 30
eps_mach = 2.220446049250313e-16
log_eps = np.log10(eps_mach)

def run_exact_simulation(pt, wt, k, var_idx):
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

# 1. Точка X*_X
pt_x = db['records_by_channel']['X']['best_pt']
wt_x = db['records_by_channel']['X']['best_wt']
k_x = db['records_by_channel']['X']['best_k']
errs_x = run_exact_simulation(pt_x, wt_x, k_x, 0)
log_e_x = np.log10(errs_x)

# 2. Точка X*_Y
pt_y = db['records_by_channel']['Y']['best_pt']
wt_y = db['records_by_channel']['Y']['best_wt']
k_y = db['records_by_channel']['Y']['best_k']
errs_y = run_exact_simulation(pt_y, wt_y, k_y, 1)
log_e_y = np.log10(errs_y)

# 3. Точка X*_Z
pt_z = db['records_by_channel']['Z']['best_pt']
wt_z = db['records_by_channel']['Z']['best_wt']
k_z = db['records_by_channel']['Z']['best_k']
errs_z = run_exact_simulation(pt_z, wt_z, k_z, 2)
log_e_z = np.log10(errs_z)

# 4. Универсальная точка X*_univ
pt_u = db['universal_point']['coordinates']
wt_u = db['universal_point']['wt']
k_u = db['universal_point']['k']
errs_u_x = run_exact_simulation(pt_u, wt_u, k_u, 0)
errs_u_y = run_exact_simulation(pt_u, wt_u, k_u, 1)
errs_u_z = run_exact_simulation(pt_u, wt_u, k_u, 2)
log_u_x = np.log10(errs_u_x)
log_u_y = np.log10(errs_u_y)
log_u_z = np.log10(errs_u_z)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.65,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

generations = np.arange(num_iters + 1)
C_X = '#1f77b4'  # Royal Blue
C_Y = '#d62728'  # Crimson Red
C_Z = '#2ca02c'  # Forest Green
C_PURPLE = '#8e44ad'

print("Построение абсолютно чистых графиков без единой текстовой выноски...")

# =========================================================================
# ГРАФИК 1: КАНАЛ X (Точка X*_X)
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(generations, log_e_x, color=C_X, marker='o', markersize=5, lw=2.2, label=r'Ошибка $\|e_i\|$ (канал $X$)')
ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach} \approx 2.22 \cdot 10^{-16}$')

ax.set_title(f'Канал связи $X$: Зависимость ошибки от итерации (точка $X^*_X$)\n$W_T = {wt_x}$ с, $K = {k_x}$', fontweight='bold', pad=10)
ax.set_xlabel('Номер итерации i', fontweight='bold')
ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
ax.set_xlim(-0.5, 30.5)
ax.set_ylim(-16.8, 0.5)
ax.set_yticks(np.arange(-16, 2, 2))
ax.legend(loc='lower left', framealpha=0.90)
plt.tight_layout()

p1 = os.path.join(PLOTS_DIR, '09_convergence_generation_x.png')
plt.savefig(p1, bbox_inches='tight')
plt.close()
print(f"-> 1. График канала X сохранен: {p1}")

# =========================================================================
# ГРАФИК 2: КАНАЛ Y (Точка X*_Y)
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(generations, log_e_y, color=C_Y, marker='s', markersize=5, lw=2.2, label=r'Ошибка $\|e_i\|$ (канал $Y$)')
ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach} \approx 2.22 \cdot 10^{-16}$')

ax.set_title(f'Канал связи $Y$: Зависимость ошибки от итерации (точка $X^*_Y$)\n$W_T = {wt_y}$ с, $K = {k_y}$', fontweight='bold', pad=10)
ax.set_xlabel('Номер итерации i', fontweight='bold')
ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
ax.set_xlim(-0.5, 30.5)
ax.set_ylim(-16.8, 0.5)
ax.set_yticks(np.arange(-16, 2, 2))
ax.legend(loc='lower left', framealpha=0.90)
plt.tight_layout()

p2 = os.path.join(PLOTS_DIR, '09_convergence_generation_y.png')
plt.savefig(p2, bbox_inches='tight')
plt.close()
print(f"-> 2. График канала Y сохранен: {p2}")

# =========================================================================
# ГРАФИК 3: КАНАЛ Z (Точка X*_Z)
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(generations, log_e_z, color=C_Z, marker='^', markersize=5, lw=2.2, label=r'Ошибка $\|e_i\|$ (канал $Z$)')
ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach} \approx 2.22 \cdot 10^{-16}$')

ax.set_title(f'Канал связи $Z$: Зависимость ошибки от итерации (точка $X^*_Z$)\n$W_T = {wt_z}$ с, $K = {k_z}$', fontweight='bold', pad=10)
ax.set_xlabel('Номер итерации i', fontweight='bold')
ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
ax.set_xlim(-0.5, 30.5)
ax.set_ylim(-16.8, 0.5)
ax.set_yticks(np.arange(-16, 2, 2))
ax.legend(loc='lower left', framealpha=0.90)
plt.tight_layout()

p3 = os.path.join(PLOTS_DIR, '09_convergence_generation_z.png')
plt.savefig(p3, bbox_inches='tight')
plt.close()
print(f"-> 3. График канала Z сохранен: {p3}")

# =========================================================================
# ГРАФИК 4: УНИВЕРСАЛЬНАЯ ТОЧКА X*_univ (Все три канала)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 5.8))
ax.plot(generations, log_u_x, color=C_X, marker='o', markersize=4.5, lw=2.0, label='Канал связи X')
ax.plot(generations, log_u_y, color=C_Y, marker='s', markersize=4.5, lw=2.0, linestyle='--', label='Канал связи Y')
ax.plot(generations, log_u_z, color=C_Z, marker='^', markersize=4.5, lw=2.0, linestyle='-.', label='Канал связи Z')
ax.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach} \approx 2.22 \cdot 10^{-16}$')

ax.set_title(f'Универсальная точка $X^*_{{univ}}$: Зависимость ошибки от итерации\n$W_T = {wt_u}$ с, $K = {k_u}$', fontweight='bold', pad=10)
ax.set_xlabel('Номер итерации i', fontweight='bold')
ax.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
ax.set_xlim(-0.5, 30.5)
ax.set_ylim(-16.8, 0.5)
ax.set_yticks(np.arange(-16, 2, 2))
ax.legend(loc='lower left', framealpha=0.90)
plt.tight_layout()

p4 = os.path.join(PLOTS_DIR, '09_convergence_generation_universal.png')
plt.savefig(p4, bbox_inches='tight')
plt.close()
print(f"-> 4. График универсальной точки сохранен: {p4}")

# Обновляем двухпанельный вариант (Рис. 9) без текстовых выносок
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.plot(generations, log_e_x, color=C_X, marker='o', markersize=4.5, lw=2.0, label=r'Канал X (точка $X^*_X$)')
ax1.plot(generations, log_e_y, color=C_Y, marker='s', markersize=4.5, lw=2.0, label=r'Канал Y (точка $X^*_Y$)')
ax1.plot(generations, log_e_z, color=C_Z, marker='^', markersize=4.5, lw=2.0, label=r'Канал Z (точка $X^*_Z$)')
ax1.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
ax1.set_title('А. Специализированные точки $X^*_X, X^*_Y, X^*_Z$', fontweight='bold')
ax1.set_xlabel('Номер итерации i', fontweight='bold')
ax1.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
ax1.set_xlim(-0.5, 30.5)
ax1.set_ylim(-16.8, 0.5)
ax1.set_yticks(np.arange(-16, 2, 2))
ax1.legend(loc='lower left', framealpha=0.90)

ax2.plot(generations, log_u_x, color=C_X, lw=2.2, label='Канал X')
ax2.plot(generations, log_u_y, color=C_Y, linestyle='--', lw=2.2, label='Канал Y')
ax2.plot(generations, log_u_z, color=C_Z, linestyle='-.', lw=2.2, label='Канал Z')
ax2.axhline(log_eps, color=C_PURPLE, linestyle='--', lw=1.5, label=r'Машинный эпсилон $\varepsilon_{mach}$')
ax2.set_title(f'Б. Универсальная точка $X^*_{{univ}}$ ($W_T = {wt_u}$ с, $K = {k_u}$)', fontweight='bold')
ax2.set_xlabel('Номер итерации i', fontweight='bold')
ax2.set_ylabel(r'$\log_{10}(\|e_i\|)$', fontweight='bold')
ax2.set_xlim(-0.5, 30.5)
ax2.set_ylim(-16.8, 0.5)
ax2.set_yticks(np.arange(-16, 2, 2))
ax2.legend(loc='lower left', framealpha=0.90)

plt.suptitle('Сходимость обратимой синхронизации по итерациям', fontweight='bold', y=0.99)
plt.tight_layout()
p_rev = os.path.join(PLOTS_DIR, '09_reversible_convergence_vs_iterations.png')
plt.savefig(p_rev, bbox_inches='tight')
plt.close()
print(f"-> 09_reversible_convergence_vs_iterations.png перерисован: {p_rev}")

print("\nВсе графики успешно обновлены!")
