#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Генерация мастер-PDF документа со всеми картами двунаправленной синхронизации
(Ошибка и Время) для сигналов X, Y, Z и графиками окон на 1M итераций.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

BASE_DIR = '/home/bobbert/Документы/leti/mnds/lab4/lab5'
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

pdf_path = os.path.join(PLOTS_DIR, "06_bidirectional_heatmaps_master_report.pdf")

pages_images = [
    # Страница 1: Двухпанельные графики (Карта ошибки + Временные ряды) для X, Y, Z
    "06_bidirectional_error_and_curves_x.png",
    "06_bidirectional_error_and_curves_y.png",
    "06_bidirectional_error_and_curves_z.png",
    # Страница 2: Тепловые карты времени для X, Y, Z
    "06_bidirectional_sync_time_heatmap_x.png",
    "06_bidirectional_sync_time_heatmap_y.png",
    "06_bidirectional_sync_time_heatmap_z.png",
    # Страница 3: 1M итераций для окон WT
    "10_reversible_window_comparison_all_3_1m_iters.png",
    "10_reversible_window_comparison_x_1m_iters.png",
    "10_reversible_window_comparison_y_1m_iters.png",
    "10_reversible_window_comparison_z_1m_iters.png"
]

with PdfPages(pdf_path) as pdf:
    # Лист 1: Сводный обзор двунаправленной синхронизации (3 ряда: X, Y, Z)
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    for row_idx, (vname, vchar) in enumerate([('X', 'x'), ('Y', 'y'), ('Z', 'z')]):
        # Загружаем сгенерированные файлы
        img_dual = Image.open(os.path.join(PLOTS_DIR, f"06_bidirectional_error_and_curves_{vchar}.png"))
        img_time = Image.open(os.path.join(PLOTS_DIR, f"06_bidirectional_sync_time_heatmap_{vchar}.png"))
        
        axes[row_idx, 0].imshow(img_dual)
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].set_title(f'Сигнал {vname}: Карта ошибки и сходимость во времени', fontweight='bold', fontsize=12)
        
        axes[row_idx, 1].imshow(img_time)
        axes[row_idx, 1].axis('off')
        axes[row_idx, 1].set_title(f'Сигнал {vname}: Тепловая карта времени синхронизации', fontweight='bold', fontsize=12)

    plt.suptitle('Двунаправленная синхронизация системы Спротта B: Сигналы X, Y, Z', fontweight='bold', fontsize=15, y=0.99)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Лист 2: Сводный обзор влияния длины окна WT на 1 000 000 итераций
    fig, ax = plt.subplots(figsize=(18, 7.5))
    img_1m = Image.open(os.path.join(PLOTS_DIR, "10_reversible_window_comparison_all_3_1m_iters.png"))
    ax.imshow(img_1m)
    ax.axis('off')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Листы 3, 4, 5: Детальные графики 1M итераций для каждого канала
    for vname, vchar in [('X', 'x'), ('Y', 'y'), ('Z', 'z')]:
        fig, ax = plt.subplots(figsize=(14, 8.5))
        img_ch_1m = Image.open(os.path.join(PLOTS_DIR, f"10_reversible_window_comparison_{vchar}_1m_iters.png"))
        ax.imshow(img_ch_1m)
        ax.axis('off')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

print(f"-> Мастер-PDF успешно создан: {pdf_path}")
