#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Генерация графиков с тепловыми картами ОШИБКИ и ВРЕМЕНИ синхронизации
для двунаправленной связи (сигналы X и Z):

1. Детальные карты в рабочем диапазоне [0, 2] x [0, 2] для сигнала Z:
   - Слева: Карта установившейся ошибки log10(||e||)
   - Справа: Карта времени синхронизации t_sync (шкала лавы inferno)
2. Аналогичные карты для сигнала X в диапазоне [0, 2] x [0, 2].
3. Сводный 4-панельный график (X и Z: Ошибка и Время).
4. Расширенный график [-2, 20] x [-2, 20] для сигнала Z с отображением
   ограниченного окна синхронизации.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit, prange

BASE_DIR = '/home/bobbert/Документы/leti/mnds/lab4/lab5'
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
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
    'grid.alpha': 0.5,
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

# =========================================================================
# 1. КАРТЫ ДЛЯ СИГНАЛА Z В ДИАПАЗОНЕ [0, 2] x [0, 2]
# =========================================================================
print("=" * 80)
print("1. РАСЧЕТ КАРТ ОШИБКИ И ВРЕМЕНИ ДЛЯ СИГНАЛА Z В ДИАПАЗОНЕ [0, 2] x [0, 2]")
print("=" * 80)

n_k_fine = 101
k_fine = np.linspace(0.0, 2.0, n_k_fine)
K2_fine, K1_fine = np.meshgrid(k_fine, k_fine)

err_z_2x2, time_z_2x2 = compute_bidir_maps(
    k_fine, k_fine, h,
    int(120.0 / h), int(20.0 / h),
    0.05, 2
)

log_err_z_plot = np.log10(np.clip(err_z_2x2.T, 1e-16, 1e2))
time_z_plot = time_z_2x2.T

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 6.8), gridspec_kw={'width_ratios': [1.08, 1.08]})

# Левая панель: Карта установившейся ошибки
ax1.set_facecolor('#2c3e50')
cf1 = ax1.contourf(K2_fine, K1_fine, log_err_z_plot, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
cs1 = ax1.contour(K2_fine, K1_fine, log_err_z_plot, levels=[-14, -10, -5, -2, 0], colors='white', alpha=0.5, linewidths=0.8)
ax1.clabel(cs1, inline=True, fontsize=8, fmt=r'$10^{%d}$')
cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
cbar1.set_label(r'$\log_{10}(\|e\|)$', fontsize=11, fontweight='bold')
ax1.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
ax1.set_title('Установившаяся ошибка (сигнал Z)', fontweight='bold', pad=10)
ax1.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax1.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
ax1.set_xlim(0, 2)
ax1.set_ylim(0, 2)

# Правая панель: Карта времени синхронизации (шкала лавы inferno)
ax2.set_facecolor('#1e272e')
t_max_z = np.nanmax(time_z_plot) if not np.all(np.isnan(time_z_plot)) else 80.0
cf2 = ax2.contourf(K2_fine, K1_fine, time_z_plot, levels=np.linspace(5, min(t_max_z, 120.0), 120), cmap='inferno', extend='both')
cbar2 = fig.colorbar(cf2, ax=ax2, pad=0.03)
cbar2.set_label(r'Время синхронизации $t_{sync}$, с', fontsize=11, fontweight='bold')
ax2.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
ax2.set_title('Время достижения синхронизации (сигнал Z)', fontweight='bold', pad=10)
ax2.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax2.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 2)

plt.suptitle('Взаимная синхронизация по сигналу Z ($K_1, K_2 \in [0, 2]$)', fontweight='bold', y=0.98, fontsize=14)
plt.tight_layout()
p_z_2x2 = os.path.join(PLOTS_DIR, "06_bidirectional_z_error_and_time_maps.png")
plt.savefig(p_z_2x2, bbox_inches='tight')
plt.close()
print(f"-> Сохранено: {p_z_2x2}")

# =========================================================================
# 2. КАРТЫ ДЛЯ СИГНАЛА X В ДИАПАЗОНЕ [0, 2] x [0, 2]
# =========================================================================
print("\n" + "=" * 80)
print("2. РАСЧЕТ КАРТ ОШИБКИ И ВРЕМЕНИ ДЛЯ СИГНАЛА X В ДИАПАЗОНЕ [0, 2] x [0, 2]")
print("=" * 80)

err_x_2x2, time_x_2x2 = compute_bidir_maps(
    k_fine, k_fine, h,
    int(80.0 / h), int(20.0 / h),
    0.05, 0
)

log_err_x_plot = np.log10(np.clip(err_x_2x2.T, 1e-16, 1e2))
time_x_plot = time_x_2x2.T

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 6.8), gridspec_kw={'width_ratios': [1.08, 1.08]})

ax1.set_facecolor('#2c3e50')
cf1 = ax1.contourf(K2_fine, K1_fine, log_err_x_plot, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
cs1 = ax1.contour(K2_fine, K1_fine, log_err_x_plot, levels=[-14, -10, -5, -2, 0], colors='white', alpha=0.5, linewidths=0.8)
ax1.clabel(cs1, inline=True, fontsize=8, fmt=r'$10^{%d}$')
cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
cbar1.set_label(r'$\log_{10}(\|e\|)$', fontsize=11, fontweight='bold')
ax1.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
ax1.set_title('Установившаяся ошибка (сигнал X)', fontweight='bold', pad=10)
ax1.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax1.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
ax1.set_xlim(0, 2)
ax1.set_ylim(0, 2)

ax2.set_facecolor('#1e272e')
cf2 = ax2.contourf(K2_fine, K1_fine, time_x_plot, levels=np.linspace(5, 75, 120), cmap='inferno', extend='both')
cbar2 = fig.colorbar(cf2, ax=ax2, pad=0.03)
cbar2.set_label(r'Время синхронизации $t_{sync}$, с', fontsize=11, fontweight='bold')
ax2.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
ax2.set_title('Время достижения синхронизации (сигнал X)', fontweight='bold', pad=10)
ax2.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax2.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 2)

plt.suptitle('Взаимная синхронизация по сигналу X ($K_1, K_2 \in [0, 2]$)', fontweight='bold', y=0.98, fontsize=14)
plt.tight_layout()
p_x_2x2 = os.path.join(PLOTS_DIR, "06_bidirectional_x_error_and_time_maps.png")
plt.savefig(p_x_2x2, bbox_inches='tight')
plt.close()
print(f"-> Сохранено: {p_x_2x2}")

# =========================================================================
# 3. СВОДНЫЙ 4-ПАНЕЛЬНЫЙ ГРАФИК (СИГНАЛЫ X И Z: ОШИБКА И ВРЕМЯ)
# =========================================================================
print("\n" + "=" * 80)
print("3. ПОСТРОЕНИЕ СВОДНОГО 4-ПАНЕЛЬНОГО ГРАФИКА (X И Z: ОШИБКА И ВРЕМЯ)")
print("=" * 80)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16.5, 14.0))

# Ряд 1: Сигнал X
ax1.set_facecolor('#2c3e50')
cf1 = ax1.contourf(K2_fine, K1_fine, log_err_x_plot, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
ax1.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
ax1.set_title('А. Установившаяся ошибка (сигнал X)', fontweight='bold', pad=10)
ax1.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax1.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
cbar1.set_label(r'$\log_{10}(\|e\|)$', fontsize=11, fontweight='bold')

ax2.set_facecolor('#1e272e')
cf2 = ax2.contourf(K2_fine, K1_fine, time_x_plot, levels=np.linspace(5, 75, 120), cmap='inferno', extend='both')
ax2.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
ax2.set_title('Б. Время синхронизации $t_{sync}$ (сигнал X)', fontweight='bold', pad=10)
ax2.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax2.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
cbar2 = fig.colorbar(cf2, ax=ax2, pad=0.03)
cbar2.set_label(r'$t_{sync}$, с', fontsize=11, fontweight='bold')

# Ряд 2: Сигнал Z
ax3.set_facecolor('#2c3e50')
cf3 = ax3.contourf(K2_fine, K1_fine, log_err_z_plot, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
ax3.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
ax3.set_title('В. Установившаяся ошибка (сигнал Z)', fontweight='bold', pad=10)
ax3.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax3.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
cbar3 = fig.colorbar(cf3, ax=ax3, pad=0.03)
cbar3.set_label(r'$\log_{10}(\|e\|)$', fontsize=11, fontweight='bold')

ax4.set_facecolor('#1e272e')
cf4 = ax4.contourf(K2_fine, K1_fine, time_z_plot, levels=np.linspace(5, min(t_max_z, 120.0), 120), cmap='inferno', extend='both')
ax4.plot([0, 2], [0, 2], color='white', linestyle='--', lw=1.3, alpha=0.7)
ax4.set_title('Г. Время синхронизации $t_{sync}$ (сигнал Z)', fontweight='bold', pad=10)
ax4.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax4.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
cbar4 = fig.colorbar(cf4, ax=ax4, pad=0.03)
cbar4.set_label(r'$t_{sync}$, с', fontsize=11, fontweight='bold')

plt.suptitle('Сравнительный анализ двунаправленной синхронизации для сигналов X и Z', fontweight='bold', y=0.99, fontsize=15)
plt.tight_layout()
p_4panel = os.path.join(PLOTS_DIR, "06_bidirectional_error_and_time_comparison_4panel.png")
plt.savefig(p_4panel, bbox_inches='tight')
plt.close()
print(f"-> Сохранено: {p_4panel}")

# =========================================================================
# 4. РАСШИРЕННЫЙ ГРАФИК [-2, 20] x [-2, 20] ДЛЯ СИГНАЛА Z
# =========================================================================
print("\n" + "=" * 80)
print("4. РАСЧЕТ РАСШИРЕННЫХ КАРТ ОШИБКИ И ВРЕМЕНИ ДЛЯ СИГНАЛА Z НА [-2, 20] x [-2, 20]")
print("=" * 80)

n_k_ext = 161
k_ext = np.linspace(-2.0, 20.0, n_k_ext)
K2_ext, K1_ext = np.meshgrid(k_ext, k_ext)

err_z_ext, time_z_ext = compute_bidir_maps(
    k_ext, k_ext, h,
    int(80.0 / h), int(20.0 / h),
    0.05, 2
)

log_err_z_ext_plot = np.log10(np.clip(err_z_ext.T, 1e-16, 1e2))
time_z_ext_plot = time_z_ext.T

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 6.8), gridspec_kw={'width_ratios': [1.08, 1.08]})

ax1.set_facecolor('#2c3e50')
cf1 = ax1.contourf(K2_ext, K1_ext, log_err_z_ext_plot, levels=np.linspace(-15, 1, 33), cmap='viridis_r', extend='both')
cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.03)
cbar1.set_label(r'$\log_{10}(\|e\|)$', fontsize=11, fontweight='bold')
ax1.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.6)
ax1.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.6)
ax1.plot([-2, 20], [-2, 20], color='white', linestyle='--', lw=1.3, alpha=0.7)
ax1.set_title('Установившаяся ошибка (сигнал Z)', fontweight='bold', pad=10)
ax1.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax1.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
ax1.set_xlim(-2, 20)
ax1.set_ylim(-2, 20)

ax2.set_facecolor('#1e272e')
cf2 = ax2.contourf(K2_ext, K1_ext, time_z_ext_plot, levels=np.linspace(5, 85, 120), cmap='inferno', extend='both')
cbar2 = fig.colorbar(cf2, ax=ax2, pad=0.03)
cbar2.set_label(r'Время синхронизации $t_{sync}$, с', fontsize=11, fontweight='bold')
ax2.axhline(0, color='white', linestyle=':', lw=1.1, alpha=0.6)
ax2.axvline(0, color='white', linestyle=':', lw=1.1, alpha=0.6)
ax2.plot([-2, 20], [-2, 20], color='white', linestyle='--', lw=1.3, alpha=0.7)
ax2.set_title('Время достижения синхронизации (сигнал Z)', fontweight='bold', pad=10)
ax2.set_xlabel('Коэффициент связи $K_2$', fontweight='bold')
ax2.set_ylabel('Коэффициент связи $K_1$', fontweight='bold')
ax2.set_xlim(-2, 20)
ax2.set_ylim(-2, 20)

plt.suptitle('Взаимная синхронизация по сигналу Z в расширенной плоскости ($K_1, K_2 \in [-2, 20]$)', fontweight='bold', y=0.98, fontsize=14)
plt.tight_layout()
p_z_ext = os.path.join(PLOTS_DIR, "06_bidirectional_z_error_and_time_extended.png")
plt.savefig(p_z_ext, bbox_inches='tight')
plt.close()
print(f"-> Сохранено: {p_z_ext}")

print("\n" + "=" * 80)
print("ВСЕ КАРТЫ ДЛЯ СИГНАЛОВ Z И X УСПЕШНО ПОСТРОЕНЫ!")
print("=" * 80)
