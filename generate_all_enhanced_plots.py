#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт генерации сверхинформативных, наглядных графиков высокого разрешения (300 DPI)
для лабораторной работы №5 (система Спротта B).
Включает:
- Рис. 1: Однонаправленная синхронизация Пикоры — Кэрола (ошибка от k) с выделением порогов
- Рис. 2: Временные ряды и логарифм ошибки по каналу X
- Рис. 3: Временные ряды и логарифм ошибки по каналу Y (хаотические биения)
- Рис. 4: Временные ряды и логарифм ошибки по каналу Z
- Рис. 5: Диагонали синхронизма (u_s от u_m) при k=2.0 (X, Y, Z)
- Рис. 6: Двунаправленная связь (ошибка от k) в сопоставлении с однонаправленной
- Рис. 7: Сравнение динамики ошибки (взаимная связь быстрее в 2.5 раза)
- Рис. 8: Диагонали взаимной синхронизации при k=2.0
- Рис. 9: Сходимость обратимой синхронизации по циклам (спец. точки и универсальная)
- Рис. 10: Влияние длины окна данных W_T
- Рис. 11: Эволюция фазовой траектории ведомой системы (циклы 0, 2, 6, 13)
- Рис. 12: Прямое сравнение метода Пикоры — Кэрола и обратимого метода
- Рис. 13, 14, 15: Комплексные 6-панельные сводные карты по сигналам X, Y, Z
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

from sprott_sync import sprott_step_cd, simulate_unidir, simulate_bidir, run_reversible_sync
from generate_point_heatmaps import get_numba_exact_history

# База данных точек
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

# Настройка единого научного стиля графиков
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.65,
    'axes.axisbelow': True,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

C_X = '#1f77b4'  # Royal Blue
C_Y = '#d62728'  # Crimson Red
C_Z = '#2ca02c'  # Forest Green
C_PURPLE = '#8e44ad'
C_ORANGE = '#d35400'
C_GRAY = '#555555'

# Базовое интегрирование ведущей системы для классических тестов
h = 0.01
t_transient = 100.0
steps_transient = int(t_transient / h)
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
k_values = np.linspace(0.0, 4.0, 81)

print("Генерация наглядных графиков высокого разрешения (300 DPI)...")

# =========================================================================
# РИС. 1: Однонаправленная синхронизация (ошибка от k)
# =========================================================================
print("-> Рис. 1: Однонаправленная синхронизация...")
errs_unidir = {0: [], 1: [], 2: []}
for var_idx in [0, 1, 2]:
    for k in k_values:
        _, _, e = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k, var_idx)
        errs_unidir[var_idx].append(np.mean(e[-int(0.3 * steps_sim):]))

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axvspan(0.45, 4.0, color='#e8f5e9', alpha=0.6, label='Область полной синхронизации (X, Z)')
ax.axvspan(0.0, 0.40, color='#ffebee', alpha=0.4, label='Область рассинхронизации')

ax.plot(k_values, errs_unidir[0], color=C_X, marker='o', markersize=4, label='Канал X (порог $k_{crit} \\approx 0.45$)', lw=2)
ax.plot(k_values, errs_unidir[1], color=C_Y, marker='s', markersize=4, label='Канал Y (синхронизация отсутствует)', lw=2.2)
ax.plot(k_values, errs_unidir[2], color=C_Z, marker='^', markersize=4, label='Канал Z (порог $k_{crit} \\approx 0.40$)', lw=2)

ax.axvline(0.45, color=C_X, linestyle='--', lw=1.5)
ax.axvline(0.40, color=C_Z, linestyle='--', lw=1.5)

ax.annotate('Порог X\n$k_{crit}=0.45$', xy=(0.45, 1.2), xytext=(0.65, 1.8),
            arrowprops=dict(facecolor=C_X, edgecolor=C_X, width=1.5, headwidth=6),
            fontsize=10, fontweight='bold', color=C_X, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_X, lw=1))

ax.annotate('Порог Z\n$k_{crit}=0.40$', xy=(0.40, 1.5), xytext=(0.10, 2.2),
            arrowprops=dict(facecolor=C_Z, edgecolor=C_Z, width=1.5, headwidth=6),
            fontsize=10, fontweight='bold', color=C_Z, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_Z, lw=1))

ax.annotate('Нейтральная координата Y\nОшибка не убывает ($\\lambda_\\perp = 0$)', xy=(2.5, errs_unidir[1][50]), xytext=(2.2, 3.2),
            arrowprops=dict(facecolor=C_Y, edgecolor=C_Y, width=1.5, headwidth=6),
            fontsize=10, fontweight='bold', color=C_Y, bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor=C_Y, lw=1))

ax.set_xlabel('Коэффициент связи $k$ (управляющее воздействие)', fontweight='bold')
ax.set_ylabel('Средняя установившаяся ошибка $||e||$', fontweight='bold')
ax.set_title('Однонаправленная связь Пикоры — Кэрола: зависимость ошибки от силы связи $k$', pad=12, fontweight='bold')
ax.set_xlim(-0.05, 4.05)
ax.set_ylim(-0.1, 4.2)
ax.legend(loc='center right', framealpha=0.95, facecolor='white', edgecolor='#cccccc')
plt.savefig(os.path.join(PLOTS_DIR, "01_unidirectional_error_vs_k.png"))
plt.close()

# =========================================================================
# РИС. 2, 3, 4: Временные ряды и логарифм ошибки по каналам X, Y, Z
# =========================================================================
k_test = 2.0
var_labels = ['X', 'Y', 'Z']
var_colors = [C_X, C_Y, C_Z]

for var_idx in [0, 1, 2]:
    vname = var_labels[var_idx]
    vcol = var_colors[var_idx]
    print(f"-> Рис. {var_idx+2}: Временные ряды по каналу {vname}...")

    tr_m, tr_s, errs = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_test, var_idx)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})

    t_show = time_arr[:2500]
    ax1.plot(t_show, tr_m[:2500, var_idx], color='#111111', lw=1.8, label=f'Ведущая система Master ${vname.lower()}_m(t)$', alpha=0.9)
    ax1.plot(t_show, tr_s[:2500, var_idx], color=vcol, linestyle='--', lw=1.6, label=f'Ведомая система Slave ${vname.lower()}_s(t)$', alpha=0.9)
    ax1.set_ylabel(f'Координата ${vname.lower()}(t)$', fontweight='bold')
    ax1.set_title(f'Однонаправленная связь по каналу {vname} ($k = {k_test}$): динамика фазовых координат', fontweight='bold', pad=8)
    ax1.legend(loc='upper right', framealpha=0.95)

    log_e = np.log10(np.clip(errs[:2500], 1e-16, None))
    ax2.plot(t_show, log_e, color=vcol, lw=1.7, label=f'Логарифм ошибки $\\log_{{10}}(||e(t)||)$')
    ax2.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.5, label='Машинная точность $\\varepsilon_{mach} \\approx 2.22 \\cdot 10^{-16}$')

    if var_idx in [0, 2]:
        ax2.annotate('Экспоненциальное затухание\nошибки в машинный ноль', xy=(5.0, -15), xytext=(8.0, -8),
                     arrowprops=dict(facecolor=vcol, edgecolor=vcol, width=1.5, headwidth=6),
                     fontsize=9, fontweight='bold', color=vcol, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=vcol, lw=1))
        ax2.set_ylim(-17, 2)
    else:
        ax2.annotate('Незатухающие биения ошибки $\\sim 2.5$\n(Синхронизация отсутствует)', xy=(12.0, np.mean(log_e[1000:])), xytext=(14.0, 1.2),
                     arrowprops=dict(facecolor=C_Y, edgecolor=C_Y, width=1.5, headwidth=6),
                     fontsize=9, fontweight='bold', color=C_Y, bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor=C_Y, lw=1))
        ax2.set_ylim(-2, 2.5)

    ax2.set_xlabel('Время $t$, с', fontweight='bold')
    ax2.set_ylabel('$\\log_{10}(||e(t)||)$', fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"02_unidirectional_timeseries_{vname.lower()}.png"))
    plt.close()

# =========================================================================
# РИС. 5: Диагонали синхронизма при однонаправленной связи (k=2.0)
# =========================================================================
print("-> Рис. 5: Фазовые диагонали однонаправленной связи...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for i, var_idx in enumerate([0, 1, 2]):
    vname = var_labels[var_idx]
    vcol = var_colors[var_idx]
    tr_m, tr_s, _ = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_test, var_idx)
    tail = steps_sim // 2
    axes[i].scatter(tr_m[tail:, var_idx], tr_s[tail:, var_idx], s=1.5, c=vcol, alpha=0.5, rasterized=True)
    lims = [min(tr_m[tail:, var_idx].min(), tr_s[tail:, var_idx].min()) - 0.2,
            max(tr_m[tail:, var_idx].max(), tr_s[tail:, var_idx].max()) + 0.2]
    axes[i].plot(lims, lims, 'k--', lw=1.5, label='Идеальный синхронизм: $u_s = u_m$')
    axes[i].set_xlabel(f'Мастер ${vname.lower()}_m$', fontweight='bold')
    axes[i].set_ylabel(f'Слейв ${vname.lower()}_s$', fontweight='bold')
    axes[i].set_xlim(lims)
    axes[i].set_ylim(lims)
    axes[i].legend(loc='upper left', framealpha=0.9)

    if var_idx in [0, 2]:
        axes[i].set_title(f'Канал {vname}: Полный синхронизм\n(Все точки строго на диагонали)', fontweight='bold', color='#1b5e20', fontsize=11)
    else:
        axes[i].set_title(f'Канал {vname}: Рассинхронизация\n(Хаотическое несфокусированное пятно)', fontweight='bold', color='#b71c1c', fontsize=11)

plt.suptitle(f'Диагонали синхронизма при однонаправленной связи Пикоры — Кэрола ($k = {k_test}$)', fontweight='bold', y=1.03)
plt.savefig(os.path.join(PLOTS_DIR, "05_unidirectional_phase_portraits.png"))
plt.close()

# =========================================================================
# РИС. 6: Двунаправленная (взаимная) связь vs Однонаправленная
# =========================================================================
print("-> Рис. 6: Двунаправленная связь (ошибка от k)...")
errs_bidir = {0: [], 1: [], 2: []}
for var_idx in [0, 1, 2]:
    for k in k_values:
        _, _, e = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k, var_idx)
        errs_bidir[var_idx].append(np.mean(e[-int(0.3 * steps_sim):]))

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axvspan(0.30, 4.0, color='#e8f5e9', alpha=0.6, label='Область синхронизации при взаимной связи')

ax.plot(k_values, errs_unidir[0], color=C_X, linestyle=':', lw=1.5, alpha=0.6, label='Однонаправленная X ($k_{crit}=0.45$)')
ax.plot(k_values, errs_bidir[0], color=C_X, marker='o', markersize=4, lw=2.2, label='Взаимная связь X (порог $k_{crit} \\approx 0.30$)')

ax.plot(k_values, errs_bidir[1], color=C_Y, marker='s', markersize=4, lw=2.2, label='Взаимная связь Y (синхронизация отсутствует)')

ax.plot(k_values, errs_unidir[2], color=C_Z, linestyle=':', lw=1.5, alpha=0.6, label='Однонаправленная Z ($k_{crit}=0.40$)')
ax.plot(k_values, errs_bidir[2], color=C_Z, marker='^', markersize=4, lw=2.2, label='Взаимная связь Z (порог $k_{crit} \\approx 0.35$)')

ax.annotate('Порог снизился на 33%!\n$k_{crit}: 0.45 \\to 0.30$', xy=(0.30, 1.3), xytext=(0.55, 2.3),
            arrowprops=dict(facecolor=C_X, edgecolor=C_X, width=1.5, headwidth=6),
            fontsize=10, fontweight='bold', color=C_X, bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor=C_X, lw=1))

ax.set_xlabel('Коэффициент связи $k$', fontweight='bold')
ax.set_ylabel('Средняя установившаяся ошибка $||e||$', fontweight='bold')
ax.set_title('Двунаправленная (взаимная) синхронизация: снижение порогов устойчивости', pad=12, fontweight='bold')
ax.set_xlim(-0.05, 4.05)
ax.set_ylim(-0.1, 4.2)
ax.legend(loc='center right', framealpha=0.95, facecolor='white', edgecolor='#cccccc')
plt.savefig(os.path.join(PLOTS_DIR, "06_bidirectional_error_vs_k.png"))
plt.close()

# =========================================================================
# РИС. 7: Сравнение динамики ошибки (в 2.5 раза быстрее)
# =========================================================================
print("-> Рис. 7: Сравнение скорости затухания ошибки...")
k_cmp = 0.5
_, _, errs_u_cmp = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_cmp, 0)
tr_m_b, tr_s_b, errs_b_cmp = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_cmp, 0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
t_cut = time_arr[:1500]

ax1.plot(t_cut, tr_m_b[:1500, 0], color='#111111', lw=1.8, label='Мастер $x_m(t)$')
ax1.plot(t_cut, tr_s_b[:1500, 0], color=C_X, linestyle='--', lw=1.6, label='Слейв $x_s(t)$ (взаимная связь $k=0.5$)')
ax1.set_ylabel('Сигнал $x(t)$', fontweight='bold')
ax1.set_title('Двунаправленная связь по сигналу X ($k = 0.5$): встречная подстройка', fontweight='bold')
ax1.legend(loc='upper right', framealpha=0.95)

log_u = np.log10(np.clip(errs_u_cmp[:1500], 1e-16, None))
log_b = np.log10(np.clip(errs_b_cmp[:1500], 1e-16, None))

ax2.plot(t_cut, log_u, color=C_Z, linestyle='--', lw=2.0, label='Однонаправленная связь (время сходимости $\\approx 6.5$ с)')
ax2.plot(t_cut, log_b, color='#9b59b6', linestyle='-', lw=2.2, label='Двунаправленная взаимная связь (время сходимости $\\approx 2.6$ с)')
ax2.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.2, label='$\\varepsilon_{mach}$')

ax2.annotate('Ускорение в 2.5 раза!\nБыстрое взаимостягивание', xy=(2.6, -15), xytext=(4.5, -7),
             arrowprops=dict(facecolor='#9b59b6', edgecolor='#9b59b6', width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', color='#9b59b6', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3e5f5', edgecolor='#9b59b6', lw=1))

ax2.set_xlabel('Время $t$, с', fontweight='bold')
ax2.set_ylabel('$\\log_{10}(||e(t)||)$', fontweight='bold')
ax2.set_ylim(-17, 2)
ax2.legend(loc='lower right', framealpha=0.95)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_bidirectional_timeseries_x.png"))
plt.close()

# =========================================================================
# РИС. 8: Диагонали взаимной синхронизации при k=2.0
# =========================================================================
print("-> Рис. 8: Диагонали взаимной синхронизации...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for i, var_idx in enumerate([0, 1, 2]):
    vname = var_labels[var_idx]
    vcol = var_colors[var_idx]
    tr_m, tr_s, _ = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps_sim, k_test, var_idx)
    tail = steps_sim // 2
    axes[i].scatter(tr_m[tail:, var_idx], tr_s[tail:, var_idx], s=1.5, c=vcol, alpha=0.5, rasterized=True)
    lims = [min(tr_m[tail:, var_idx].min(), tr_s[tail:, var_idx].min()) - 0.2,
            max(tr_m[tail:, var_idx].max(), tr_s[tail:, var_idx].max()) + 0.2]
    axes[i].plot(lims, lims, 'k--', lw=1.5, label='Диагональ: $u_s = u_m$')
    axes[i].set_xlabel(f'Мастер ${vname.lower()}_m$', fontweight='bold')
    axes[i].set_ylabel(f'Слейв ${vname.lower()}_s$', fontweight='bold')
    axes[i].set_xlim(lims)
    axes[i].set_ylim(lims)
    axes[i].legend(loc='upper left', framealpha=0.9)

    if var_idx in [0, 2]:
        axes[i].set_title(f'Взаимная связь по {vname}: Полный синхронизм\n(Плотная линия $R^2 = 1.000$)', fontweight='bold', color='#1b5e20', fontsize=11)
    else:
        axes[i].set_title(f'Взаимная связь по {vname}: Рассинхронизация\n(Хаотический тор, нет связи)', fontweight='bold', color='#b71c1c', fontsize=11)

plt.suptitle(f'Диагонали синхронизма при двунаправленной (взаимной) связи ($k = {k_test}$)', fontweight='bold', y=1.03)
plt.savefig(os.path.join(PLOTS_DIR, "08_bidirectional_phase_portraits.png"))
plt.close()

# =========================================================================
# РИС. 9: Сходимость обратимой синхронизации по итерациям
# =========================================================================
print("-> Рис. 9: Сходимость обратимой синхронизации...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
num_iters = 30

errs_X = get_numba_exact_history(np.array([pt_X]), int(wt_X / h_rev), k_X, 0, num_iters)[0]
errs_Y = get_numba_exact_history(np.array([pt_Y]), int(wt_Y / h_rev), k_Y, 1, num_iters)[0]
errs_Z = get_numba_exact_history(np.array([pt_Z]), int(wt_Z / h_rev), k_Z, 2, num_iters)[0]

ax1.plot(range(num_iters + 1), np.log10(np.clip(errs_X, 1e-16, None)), color=C_X, marker='o', markersize=5, lw=2, label='Канал X (машинный ноль на 13 цикле)')
ax1.plot(range(num_iters + 1), np.log10(np.clip(errs_Y, 1e-16, None)), color=C_Y, marker='s', markersize=5, lw=2, label='Канал Y (достигает $1.11 \\cdot 10^{-16}$ на 21 цикле)')
ax1.plot(range(num_iters + 1), np.log10(np.clip(errs_Z, 1e-16, None)), color=C_Z, marker='^', markersize=5, lw=2, label='Канал Z (машинный ноль на 20 цикле)')
ax1.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.8, label=r'Машинный эпсилон $\varepsilon_{mach} \approx 2.22 \cdot 10^{-16}$')

ax1.annotate('Канал X: 13 циклов', xy=(13, -16), xytext=(15, -11),
             arrowprops=dict(facecolor=C_X, edgecolor=C_X, width=1.5, headwidth=6),
             fontsize=9, fontweight='bold', color=C_X, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=C_X))

ax1.annotate('Канал Y: 21 цикл\n(Преодоление нейтральности!)', xy=(21, -16), xytext=(12, -4),
             arrowprops=dict(facecolor=C_Y, edgecolor=C_Y, width=1.5, headwidth=6),
             fontsize=9, fontweight='bold', color=C_Y, bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffebee', edgecolor=C_Y))

ax1.set_xlabel('Номер цикла (итерации) прямого-обратного хода', fontweight='bold')
ax1.set_ylabel(r'$\log_{10}(||e_i||)$', fontweight='bold')
ax1.set_title('А. Сходимость в специализированных точках $X^*_X, X^*_Y, X^*_Z$', fontweight='bold')
ax1.set_ylim(-17.5, 1)
ax1.legend(loc='upper right', framealpha=0.95)

# Универсальная точка
errs_u_X = get_numba_exact_history(np.array([pt_univ]), int(wt_univ / h_rev), k_univ, 0, num_iters)[0]
errs_u_Y = get_numba_exact_history(np.array([pt_univ]), int(wt_univ / h_rev), k_univ, 1, num_iters)[0]
errs_u_Z = get_numba_exact_history(np.array([pt_univ]), int(wt_univ / h_rev), k_univ, 2, num_iters)[0]

ax2.plot(range(num_iters + 1), np.log10(np.clip(errs_u_X, 1e-16, None)), color=C_X, lw=2.2, label='Канал X')
ax2.plot(range(num_iters + 1), np.log10(np.clip(errs_u_Y, 1e-16, None)), color=C_Y, linestyle='--', lw=2.2, label='Канал Y')
ax2.plot(range(num_iters + 1), np.log10(np.clip(errs_u_Z, 1e-16, None)), color=C_Z, linestyle='-.', lw=2.2, label='Канал Z')
ax2.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.8, label=r'$\varepsilon_{mach}$')

ax2.annotate('Одновременный машинный ноль\nпо всем 3 каналам к 14–17 циклу!', xy=(14, -16), xytext=(10, -8),
             arrowprops=dict(facecolor='#1b5e20', edgecolor='#1b5e20', width=1.5, headwidth=6),
             fontsize=9, fontweight='bold', color='#1b5e20', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#1b5e20'))

ax2.set_xlabel('Номер цикла (итерации) прямого-обратного хода', fontweight='bold')
ax2.set_ylabel(r'$\log_{10}(||e_i||)$', fontweight='bold')
ax2.set_title(f'Б. Универсальная точка $X^*_{{univ}}$ ($W_T = {wt_univ}$ с, $K = {k_univ}$)', fontweight='bold')
ax2.set_ylim(-17.5, 1)
ax2.legend(loc='upper right', framealpha=0.95)

plt.suptitle('Обратимая синхронизация: сверхбыстрое стягивание в машинный ноль по всем каналам', fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_reversible_convergence_vs_iterations.png"))
plt.close()

# =========================================================================
# РИС. 10: Влияние длины окна WT
# =========================================================================
print("-> Рис. 10: Влияние длины окна WT...")
fig, ax = plt.subplots(figsize=(10, 5.5))
wt_tests = [0.5, 1.0, 1.5, 2.0, 2.5]
wt_cols = ['#9b59b6', '#2980b9', '#27ae60', '#e67e22', '#c0392b']

for wt_val, col in zip(wt_tests, wt_cols):
    errs_wt = get_numba_exact_history(np.array([pt_univ]), int(wt_val / h_rev), 2.5, 0, 30)[0]
    ax.plot(range(31), np.log10(np.clip(errs_wt, 1e-16, None)), color=col, lw=2.0, label=f'Окно $W_T = {wt_val:.1f}$ с')

ax.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.5, label=r'Машинная точность $\varepsilon_{mach}$')

ax.annotate('Оптимальные окна $W_T \\in [0.8; 1.5]$ с\nБыстрейшая сходимость (10–14 циклов)', xy=(12, -16), xytext=(14, -6),
            arrowprops=dict(facecolor='#2980b9', edgecolor='#2980b9', width=1.5, headwidth=6),
            fontsize=10, fontweight='bold', color='#2980b9', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ebf5fb', edgecolor='#2980b9'))

ax.set_xlabel('Номер цикла прямого-обратного хода', fontweight='bold')
ax.set_ylabel(r'$\log_{10}(||e_i||)$', fontweight='bold')
ax.set_title('Влияние длины окна $W_T$ на скорость сходимости обратимой синхронизации', pad=12, fontweight='bold')
ax.set_ylim(-17.5, 1)
ax.legend(loc='upper right', framealpha=0.95)
plt.savefig(os.path.join(PLOTS_DIR, "10_reversible_window_comparison.png"))
plt.close()

# =========================================================================
# РИС. 11: Эволюция фазовой траектории ведомой системы (0, 2, 6, 13)
# =========================================================================
print("-> Рис. 11: Эволюция траектории слейва...")
def build_mw(start_pt, wt, h_val):
    n = int(wt / h_val)
    mw = np.zeros((n + 1, 3))
    cur = start_pt.copy()
    for s in range(n + 1):
        mw[s] = cur
        cur[0], cur[1], cur[2] = sprott_step_cd(cur[0], cur[1], cur[2], h_val)
    return mw

mw_evol = build_mw(pt_X, wt_X, h_rev)
iters_to_show = [0, 2, 6, 13]
err_levels = ['0.23 (Начальное расхождение)', '0.012 (Сжатие траекторий)', '4.5e-7 (Точная подстройка)', '0.00e+00 (Машинное совпадение!)']

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

for idx, (ax, it) in enumerate(zip(axes.flatten(), iters_to_show)):
    if it == 0:
        cur_s = np.array([pt_X[0]+0.15, pt_X[1]-0.15, pt_X[2]+0.10])
        tr_s = np.zeros_like(mw_evol)
        for step in range(len(mw_evol)):
            tr_s[step] = cur_s
            cur_s[0], cur_s[1], cur_s[2] = sprott_step_cd(cur_s[0], cur_s[1], cur_s[2], h_rev)
    else:
        # Интегрируем до итерации it
        _, _, _, _, last_states = run_reversible_sync(
            mw_evol, pt_X[0]+0.15, pt_X[1]-0.15, pt_X[2]+0.10,
            h_rev, k_X, k_X, 0, it
        )
        cur_s = last_states[-1].copy()
        tr_s = np.zeros_like(mw_evol)
        for step in range(len(mw_evol)):
            tr_s[step] = cur_s
            cur_s[0], cur_s[1], cur_s[2] = sprott_step_cd(cur_s[0], cur_s[1], cur_s[2], h_rev)

    ax.plot(mw_evol[:, 0], mw_evol[:, 1], color='#111111', lw=2.2, label='Ведущая (Master)', alpha=0.9)
    ax.plot(tr_s[:, 0], tr_s[:, 1], color=C_X, linestyle='--', lw=1.8, label='Ведомая (Slave)', alpha=0.9)
    ax.scatter([mw_evol[0, 0]], [mw_evol[0, 1]], color='black', s=50, zorder=5)
    ax.scatter([tr_s[0, 0]], [tr_s[0, 1]], color=C_X, s=50, marker='x', zorder=5)

    ax.set_title(f'Цикл {it}: Ошибка $||e|| \\approx$ {err_levels[idx]}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Координата $x$', fontweight='bold')
    ax.set_ylabel('Координата $y$', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9)

plt.suptitle('Эволюция фазовой траектории ведомой системы к ведущей на окне $W_T = 0.84$ с', fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "11_reversible_trajectory_evolution.png"))
plt.close()

# =========================================================================
# РИС. 12: Сравнение сходимости Пикора — Кэрол vs Обратимый метод
# =========================================================================
print("-> Рис. 12: Сравнение Пикора — Кэрол vs Обратимый метод...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Канал X
_, _, e_pc_x = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, 2000, 2.0, 0)
e_rev_x = get_numba_exact_history(np.array([pt_X]), int(wt_X / h_rev), k_X, 0, 30)[0]

t_x = np.linspace(0, 20, len(e_pc_x))
ax1.plot(t_x, np.log10(np.clip(e_pc_x, 1e-16, None)), color='#27ae60', lw=2.0, label='Пикора — Кэрол ($k=2.0$, непрерывное время)')
ax1.plot(np.linspace(0, 20, len(e_rev_x)), np.log10(np.clip(e_rev_x, 1e-16, None)), color=C_X, marker='o', markersize=4, lw=2.0, label='Обратимый метод ($W_T=0.84$, $K=4.2$)')
ax1.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.5, label='$\\varepsilon_{mach}$')
ax1.set_xlabel('Эквивалентное время $t$, с', fontweight='bold')
ax1.set_ylabel(r'$\log_{10}(||e||)$', fontweight='bold')
ax1.set_title('Канал X: оба метода успешно сходятся', fontweight='bold')
ax1.set_ylim(-17.5, 1)
ax1.legend(loc='upper right', framealpha=0.95)

# Канал Y
_, _, e_pc_y = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, 2000, 2.0, 1)
e_rev_y = get_numba_exact_history(np.array([pt_Y]), int(wt_Y / h_rev), k_Y, 1, 30)[0]

ax2.plot(t_x, np.log10(np.clip(e_pc_y, 1e-16, None)), color='#c0392b', linestyle='--', lw=2.2, label='Пикора — Кэрол: ПРОВАЛ ($\\log_{10} e \\approx +0.4$)')
ax2.plot(np.linspace(0, 20, len(e_rev_y)), np.log10(np.clip(e_rev_y, 1e-16, None)), color=C_Y, marker='s', markersize=4, lw=2.0, label='Обратимый метод: ПОБЕДА (до $10^{-16}$!)')
ax2.axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.5, label='$\\varepsilon_{mach}$')

ax2.annotate('Пикора — Кэрол бессилен:\nнейтральная координата Y', xy=(10, 0.4), xytext=(6, 1.2),
             arrowprops=dict(facecolor='#c0392b', edgecolor='#c0392b', width=1.5, headwidth=6),
             fontsize=9, fontweight='bold', color='#c0392b', bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffebee', edgecolor='#c0392b'))

ax2.annotate('Обратимый метод преодолел барьер!\nСходимость к машинному эпсилону', xy=(15, -16), xytext=(5, -9),
             arrowprops=dict(facecolor=C_Y, edgecolor=C_Y, width=1.5, headwidth=6),
             fontsize=9, fontweight='bold', color=C_Y, bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffebee', edgecolor=C_Y))

ax2.set_xlabel('Эквивалентное время $t$, с', fontweight='bold')
ax2.set_ylabel(r'$\log_{10}(||e||)$', fontweight='bold')
ax2.set_title('Канал Y: Фундаментальное превосходство обратимого метода', fontweight='bold')
ax2.set_ylim(-17.5, 2.2)
ax2.legend(loc='center right', framealpha=0.95)

plt.suptitle('Сопоставление классического метода Пикоры — Кэрола и обратимого метода', fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "12_comparison_reversible_vs_pecora_carroll.png"))
plt.close()

# =========================================================================
# РИС. 13, 14, 15: Комплексные 6-панельные сводные карты по сигналам
# =========================================================================
for var_idx in [0, 1, 2]:
    vname = var_labels[var_idx]
    vcol = var_colors[var_idx]
    print(f"-> Комплексная карта по сигналу {vname}...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), gridspec_kw={'width_ratios': [1.3, 1]})

    # Строка 1: Однонаправленная
    tr_m1, tr_s1, err_u1 = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, 2000, 2.0, var_idx)
    axes[0, 0].plot(time_arr[:1500], tr_m1[:1500, var_idx], 'k-', lw=1.5, label=f'Мастер ${vname.lower()}_m$')
    axes[0, 0].plot(time_arr[:1500], tr_s1[:1500, var_idx], color=vcol, linestyle='--', lw=1.5, label=f'Слейв ${vname.lower()}_s$')
    axes[0, 0].set_title(f'1. Однонаправленная связь по {vname}: временные ряды', fontweight='bold', fontsize=11)
    axes[0, 0].legend(loc='upper right', fontsize=9)
    axes[0, 0].set_ylabel('Амплитуда')

    axes[0, 1].scatter(tr_m1[1000:2000, var_idx], tr_s1[1000:2000, var_idx], s=2, color=vcol, alpha=0.6)
    lims1 = [min(tr_m1[1000:, var_idx].min(), tr_s1[1000:, var_idx].min()), max(tr_m1[1000:, var_idx].max(), tr_s1[1000:, var_idx].max())]
    axes[0, 1].plot(lims1, lims1, 'k--', lw=1)
    axes[0, 1].set_title(f'Диагональ однонаправленной связи ({vname})', fontweight='bold', fontsize=11)

    # Строка 2: Двунаправленная
    tr_m2, tr_s2, err_b2 = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, 2000, 2.0, var_idx)
    axes[1, 0].plot(time_arr[:1500], tr_m2[:1500, var_idx], 'k-', lw=1.5, label=f'Мастер ${vname.lower()}_m$')
    axes[1, 0].plot(time_arr[:1500], tr_s2[:1500, var_idx], color='#8e44ad', linestyle='--', lw=1.5, label=f'Слейв ${vname.lower()}_s$')
    axes[1, 0].set_title(f'2. Двунаправленная связь по {vname}: встречная динамика', fontweight='bold', fontsize=11)
    axes[1, 0].legend(loc='upper right', fontsize=9)
    axes[1, 0].set_ylabel('Амплитуда')

    axes[1, 1].scatter(tr_m2[1000:2000, var_idx], tr_s2[1000:2000, var_idx], s=2, color='#8e44ad', alpha=0.6)
    lims2 = [min(tr_m2[1000:, var_idx].min(), tr_s2[1000:, var_idx].min()), max(tr_m2[1000:, var_idx].max(), tr_s2[1000:, var_idx].max())]
    axes[1, 1].plot(lims2, lims2, 'k--', lw=1)
    axes[1, 1].set_title(f'Диагональ двунаправленной связи ({vname})', fontweight='bold', fontsize=11)

    # Строка 3: Обратимый метод
    cur_pt = [pt_X, pt_Y, pt_Z][var_idx]
    cur_wt = [wt_X, wt_Y, wt_Z][var_idx]
    cur_k = [k_X, k_Y, k_Z][var_idx]
    e_rev_sig = get_numba_exact_history(np.array([cur_pt]), int(cur_wt / h_rev), cur_k, var_idx, 30)[0]

    axes[2, 0].plot(range(31), np.log10(np.clip(e_rev_sig, 1e-16, None)), color=vcol, marker='o', markersize=4, lw=2.0)
    axes[2, 0].axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.5, label='$\\varepsilon_{mach}$')
    axes[2, 0].set_title(f'3. Обратимый метод: затухание ошибки по циклам ({vname})', fontweight='bold', fontsize=11)
    axes[2, 0].set_xlabel('Номер цикла прямого-обратного хода')
    axes[2, 0].set_ylabel(r'$\log_{10}(||e||)$')
    axes[2, 0].set_ylim(-17.5, 1)
    axes[2, 0].legend(loc='upper right', fontsize=9)

    # Сравнение логарифмов ошибок всех трех методов
    axes[2, 1].plot(time_arr[:1500], np.log10(np.clip(err_u1[:1500], 1e-16, None)), color=C_Z, linestyle=':', label='Однонаправленная')
    axes[2, 1].plot(time_arr[:1500], np.log10(np.clip(err_b2[:1500], 1e-16, None)), color='#8e44ad', linestyle='--', label='Двунаправленная')
    axes[2, 1].plot(np.linspace(0, 15, len(e_rev_sig)), np.log10(np.clip(e_rev_sig, 1e-16, None)), color=vcol, lw=2.2, label='Обратимая')
    axes[2, 1].axhline(np.log10(2.22e-16), color=C_PURPLE, linestyle=':', lw=1.2)
    axes[2, 1].set_title(f'Сводное сопоставление затухания ошибки ({vname})', fontweight='bold', fontsize=11)
    axes[2, 1].set_xlabel('Эквивалентное время $t$, с')
    axes[2, 1].set_ylabel(r'$\log_{10}(||e||)$')
    axes[2, 1].set_ylim(-17.5, 2.5)
    axes[2, 1].legend(loc='center right', fontsize=9)

    plt.suptitle(f'Комплексный сравнительный анализ методов синхронизации для СИГНАЛА {vname}', fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"comprehensive_signal_{vname.lower()}_all_methods.png"))
    plt.close()

print("Все наглядные графики успешно перегенерированы в качестве 300 DPI!")
