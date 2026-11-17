#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Интерактивное графическое приложение (PyQt5 + Matplotlib)
для исследования хаотической синхронизации системы Спротта B:
- Однонаправленная синхронизация Пикоры — Кэрола (Pecora-Carroll);
- Двунаправленная (взаимная) синхронизация;
- Быстрая обратимая синхронизация (Time-Reversible Synchronization).
"""

import sys
import os
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sprott_sync import (
    sprott_step_cd,
    simulate_unidir,
    simulate_bidir,
    run_reversible_sync
)


class SprottSyncApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("МНДС — Лабораторная работа №5: Синхронизация системы Спротта B")
        self.resize(1200, 800)

        self._init_ui()
        self._set_defaults()
        self.run_simulation()

    def _init_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        # ----------------- Левая панель параметров -----------------
        control_panel = QtWidgets.QWidget()
        control_panel.setFixedWidth(340)
        control_layout = QtWidgets.QVBoxLayout(control_panel)

        # Группа 1: Режим синхронизации
        group_mode = QtWidgets.QGroupBox("Режим синхронизации")
        layout_mode = QtWidgets.QVBoxLayout(group_mode)
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems([
            "Однонаправленная (Pecora-Carroll)",
            "Двунаправленная (взаимная)",
            "Обратимая (Time-Reversible)"
        ])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        layout_mode.addWidget(self.combo_mode)

        layout_mode.addWidget(QtWidgets.QLabel("Канал (сигнал) связи:"))
        self.combo_var = QtWidgets.QComboBox()
        self.combo_var.addItems(["Сигнал x", "Сигнал y", "Сигнал z"])
        layout_mode.addWidget(self.combo_var)
        control_layout.addWidget(group_mode)

        # Группа 2: Параметры связи
        self.group_params = QtWidgets.QGroupBox("Параметры управления")
        layout_params = QtWidgets.QFormLayout(self.group_params)

        self.spin_k = QtWidgets.QDoubleSpinBox()
        self.spin_k.setRange(0.0, 50.0)
        self.spin_k.setSingleStep(0.1)
        self.spin_k.setValue(2.0)
        layout_params.addRow("Коэффициент K:", self.spin_k)

        self.spin_h = QtWidgets.QDoubleSpinBox()
        self.spin_h.setRange(0.001, 0.1)
        self.spin_h.setSingleStep(0.002)
        self.spin_h.setDecimals(4)
        self.spin_h.setValue(0.01)
        layout_params.addRow("Шаг интеграции h:", self.spin_h)

        self.spin_time = QtWidgets.QDoubleSpinBox()
        self.spin_time.setRange(1.0, 200.0)
        self.spin_time.setSingleStep(5.0)
        self.spin_time.setValue(30.0)
        self.lbl_time = QtWidgets.QLabel("Время T (с):")
        layout_params.addRow(self.lbl_time, self.spin_time)

        # Доп. параметры для обратимой синхронизации
        self.spin_wt = QtWidgets.QDoubleSpinBox()
        self.spin_wt.setRange(0.1, 10.0)
        self.spin_wt.setSingleStep(0.1)
        self.spin_wt.setValue(1.0)
        self.lbl_wt = QtWidgets.QLabel("Окно данных WT (с):")
        layout_params.addRow(self.lbl_wt, self.spin_wt)

        self.spin_iters = QtWidgets.QSpinBox()
        self.spin_iters.setRange(1, 200)
        self.spin_iters.setValue(50)
        self.lbl_iters = QtWidgets.QLabel("Число итераций:")
        layout_params.addRow(self.lbl_iters, self.spin_iters)

        control_layout.addWidget(self.group_params)

        # Группа 3: Начальные смещения ведомой системы
        group_offsets = QtWidgets.QGroupBox("Начальное смещение Slave (Δ)")
        layout_offsets = QtWidgets.QFormLayout(group_offsets)

        self.spin_dx = QtWidgets.QDoubleSpinBox()
        self.spin_dx.setRange(-5.0, 5.0)
        self.spin_dx.setValue(0.5)
        layout_offsets.addRow("Δx_0:", self.spin_dx)

        self.spin_dy = QtWidgets.QDoubleSpinBox()
        self.spin_dy.setRange(-5.0, 5.0)
        self.spin_dy.setValue(-0.4)
        layout_offsets.addRow("Δy_0:", self.spin_dy)

        self.spin_dz = QtWidgets.QDoubleSpinBox()
        self.spin_dz.setRange(-5.0, 5.0)
        self.spin_dz.setValue(0.3)
        layout_offsets.addRow("Δz_0:", self.spin_dz)

        control_layout.addWidget(group_offsets)

        # Кнопки управления
        btn_run = QtWidgets.QPushButton("Запустить расчет")
        btn_run.setFixedHeight(38)
        btn_run.setFont(QtGui.QFont("SansSerif", 10, QtGui.QFont.Bold))
        btn_run.setStyleSheet("background-color: #2ea44f; color: white; border-radius: 4px;")
        btn_run.clicked.connect(self.run_simulation)
        control_layout.addWidget(btn_run)

        # Информационный блок
        self.info_box = QtWidgets.QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setFixedHeight(160)
        self.info_box.setStyleSheet("background-color: #f6f8fa; border: 1px solid #d0d7de; font-size: 11px;")
        control_layout.addWidget(self.info_box)

        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        # ----------------- Правая панель графиков -----------------
        self.tabs = QtWidgets.QTabWidget()

        # Таб 1: Временные ряды и логарифмическая ошибка
        self.tab_ts = QtWidgets.QWidget()
        layout_ts = QtWidgets.QVBoxLayout(self.tab_ts)
        self.fig_ts = Figure(figsize=(7, 6))
        self.canvas_ts = FigureCanvas(self.fig_ts)
        self.toolbar_ts = NavigationToolbar(self.canvas_ts, self.tab_ts)
        layout_ts.addWidget(self.toolbar_ts)
        layout_ts.addWidget(self.canvas_ts)
        self.tabs.addTab(self.tab_ts, "Временные ряды и Ошибка")

        # Таб 2: 3D фазовое пространство
        self.tab_3d = QtWidgets.QWidget()
        layout_3d = QtWidgets.QVBoxLayout(self.tab_3d)
        self.fig_3d = Figure(figsize=(7, 6))
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self.tab_3d)
        layout_3d.addWidget(self.toolbar_3d)
        layout_3d.addWidget(self.canvas_3d)
        self.tabs.addTab(self.tab_3d, "3D Аттрактор (Master/Slave)")

        # Таб 3: Плоскость синхронизации (u_s vs u_m)
        self.tab_diag = QtWidgets.QWidget()
        layout_diag = QtWidgets.QVBoxLayout(self.tab_diag)
        self.fig_diag = Figure(figsize=(7, 6))
        self.canvas_diag = FigureCanvas(self.fig_diag)
        self.toolbar_diag = NavigationToolbar(self.canvas_diag, self.tab_diag)
        layout_diag.addWidget(self.toolbar_diag)
        layout_diag.addWidget(self.canvas_diag)
        self.tabs.addTab(self.tab_diag, "Диагональ синхронизации")

        main_layout.addWidget(self.tabs)

    def _set_defaults(self):
        self._on_mode_changed(0)

    def _on_mode_changed(self, idx):
        is_rev = (idx == 2)
        self.lbl_time.setVisible(not is_rev)
        self.spin_time.setVisible(not is_rev)
        self.lbl_wt.setVisible(is_rev)
        self.spin_wt.setVisible(is_rev)
        self.lbl_iters.setVisible(is_rev)
        self.spin_iters.setVisible(is_rev)

    def run_simulation(self):
        mode_idx = self.combo_mode.currentIndex()
        var_idx = self.combo_var.currentIndex()
        var_name = ['x', 'y', 'z'][var_idx]
        K = self.spin_k.value()
        h = self.spin_h.value()

        dx0 = self.spin_dx.value()
        dy0 = self.spin_dy.value()
        dz0 = self.spin_dz.value()

        # Инициализация точки на аттракторе для Master
        xm, ym, zm = 0.2, 0.2, 0.2
        for _ in range(int(100.0 / h)):
            xm, ym, zm = sprott_step_cd(xm, ym, zm, h)

        x0_m, y0_m, z0_m = xm, ym, zm
        x0_s = x0_m + dx0
        y0_s = y0_m + dy0
        z0_s = z0_m + dz0

        init_err = np.sqrt(dx0**2 + dy0**2 + dz0**2)

        if mode_idx in [0, 1]:
            # Непрерывная синхронизация
            t_total = self.spin_time.value()
            steps = int(t_total / h)
            time_arr = np.linspace(0, t_total, steps)

            if mode_idx == 0:
                traj_m, traj_s, errs = simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps, K, var_idx)
                mode_str = "Однонаправленная (Pecora-Carroll)"
            else:
                traj_m, traj_s, errs = simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps, K, var_idx)
                mode_str = "Двунаправленная (взаимная)"

            tail = int(0.2 * steps)
            final_err = np.mean(errs[-tail:])
            is_sync = (final_err < 1e-4)

            # Вывод информации
            status_text = "СИНХРОНИЗИРОВАНО" if is_sync else "НЕТ СИНХРОНИЗАЦИИ"
            self.info_box.setHtml(f"""
            <b>Режим:</b> {mode_str}<br>
            <b>Канал связи:</b> {var_name}<br>
            <b>Коэффициент связи K:</b> {K:.2f}<br>
            <b>Шаг h:</b> {h:.4f}<br>
            <b>Начальная ошибка:</b> {init_err:.4f}<br>
            <b>Финальная средняя ошибка:</b> {final_err:.4e}<br>
            <b>Статус:</b> <font color="{'green' if is_sync else 'red'}"><b>{status_text}</b></font>
            """)

            # Отрисовка Таб 1: Временные ряды
            self.fig_ts.clear()
            ax1 = self.fig_ts.add_subplot(2, 1, 1)
            ax1.plot(time_arr, traj_m[:, var_idx], 'b-', label=f'Master {var_name}_m', alpha=0.85)
            ax1.plot(time_arr, traj_s[:, var_idx], 'r--', label=f'Slave {var_name}_s', alpha=0.85)
            ax1.set_ylabel(f'Координата {var_name}')
            ax1.set_title(f'{mode_str}: Связь по сигналу {var_name}')
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend(loc='upper right')

            ax2 = self.fig_ts.add_subplot(2, 1, 2, sharex=ax1)
            safe_errs = np.clip(errs, 1e-16, None)
            ax2.plot(time_arr, np.log10(safe_errs), 'k-', alpha=0.85)
            ax2.set_xlabel('Время t, с')
            ax2.set_ylabel('log10(||e(t)||)')
            ax2.grid(True, linestyle='--', alpha=0.5)
            self.fig_ts.tight_layout()
            self.canvas_ts.draw()

            # Отрисовка Таб 2: 3D Аттрактор
            self.fig_3d.clear()
            ax3 = self.fig_3d.add_subplot(1, 1, 1, projection='3d')
            ax3.plot(traj_m[:, 0], traj_m[:, 1], traj_m[:, 2], 'b-', label='Master', alpha=0.7, linewidth=1.0)
            ax3.plot(traj_s[:, 0], traj_s[:, 1], traj_s[:, 2], 'r--', label='Slave', alpha=0.7, linewidth=1.0)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.set_title('3D Фазовые траектории')
            ax3.legend()
            self.fig_3d.tight_layout()
            self.canvas_3d.draw()

            # Отрисовка Таб 3: Диагональ
            self.fig_diag.clear()
            ax_diag = self.fig_diag.add_subplot(1, 1, 1)
            start_idx = steps // 2
            ax_diag.scatter(traj_m[start_idx:, var_idx], traj_s[start_idx:, var_idx], s=1, c='navy', alpha=0.6)
            min_val = min(traj_m[start_idx:, var_idx].min(), traj_s[start_idx:, var_idx].min())
            max_val = max(traj_m[start_idx:, var_idx].max(), traj_s[start_idx:, var_idx].max())
            ax_diag.plot([min_val, max_val], [min_val, max_val], 'r--', label='u_s = u_m (Синхронизм)')
            ax_diag.set_xlabel(f'{var_name}_m')
            ax_diag.set_ylabel(f'{var_name}_s')
            ax_diag.set_title(f'Диагональ синхронизации {var_name}_s от {var_name}_m')
            ax_diag.grid(True, linestyle='--', alpha=0.5)
            ax_diag.legend()
            self.fig_diag.tight_layout()
            self.canvas_diag.draw()

        else:
            # Обратимая синхронизация (Time-Reversible)
            WT = self.spin_wt.value()
            num_iters = self.spin_iters.value()
            N_window = int(WT / h)

            master_window = np.zeros((N_window + 1, 3))
            xm_t, ym_t, zm_t = x0_m, y0_m, z0_m
            for i in range(N_window + 1):
                master_window[i] = [xm_t, ym_t, zm_t]
                xm_t, ym_t, zm_t = sprott_step_cd(xm_t, ym_t, zm_t, h)

            iter_errs, final_xs, final_ys, final_zs, last_fwd_traj = run_reversible_sync(
                master_window, x0_s, y0_s, z0_s, h, K, K, var_idx, num_iters
            )

            final_err = iter_errs[-1]
            delta_log = np.log10(np.clip(final_err, 1e-16, None)) - np.log10(iter_errs[0])

            self.info_box.setHtml(f"""
            <b>Режим:</b> Быстрая обратимая (Time-Reversible)<br>
            <b>Канал связи:</b> {var_name}<br>
            <b>Окно данных WT:</b> {WT:.2f} с ({N_window} точек)<br>
            <b>Число итераций:</b> {num_iters}<br>
            <b>Коэффициенты K+, K-:</b> {K:.2f}<br>
            <b>Начальная ошибка ||e_0||:</b> {iter_errs[0]:.4f}<br>
            <b>Финальная ошибка ||e_end||:</b> {final_err:.4e}<br>
            <b>Δlog10(E):</b> {delta_log:.2f} пор.
            """)

            # Отрисовка Таб 1: Сходимость по итерациям
            self.fig_ts.clear()
            ax1 = self.fig_ts.add_subplot(2, 1, 1)
            t_win = np.linspace(0, WT, N_window + 1)
            ax1.plot(t_win, master_window[:, var_idx], 'b-', label=f'Master {var_name}_m', linewidth=2.0)
            ax1.plot(t_win, last_fwd_traj[:, var_idx], 'r--', label=f'Slave {var_name}_s (итерация {num_iters})', linewidth=1.8)
            ax1.set_ylabel(f'Координата {var_name}')
            ax1.set_title(f'Восстановление сигнала на окне WT = {WT} с')
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend(loc='upper right')

            ax2 = self.fig_ts.add_subplot(2, 1, 2)
            dlog = np.log10(np.clip(iter_errs, 1e-16, None)) - np.log10(iter_errs[0])
            ax2.plot(range(num_iters + 1), dlog, 'm-o', markersize=3, alpha=0.9)
            ax2.set_xlabel('Номер итерации i')
            ax2.set_ylabel('Δ log10(E)')
            ax2.set_title('Логарифмическая ошибка сходимости по итерациям')
            ax2.grid(True, linestyle='--', alpha=0.5)
            self.fig_ts.tight_layout()
            self.canvas_ts.draw()

            # Отрисовка Таб 2: 3D Окно
            self.fig_3d.clear()
            ax3 = self.fig_3d.add_subplot(1, 1, 1, projection='3d')
            ax3.plot(master_window[:, 0], master_window[:, 1], master_window[:, 2], 'b-', label='Master (окно)', linewidth=2.0)
            ax3.plot(last_fwd_traj[:, 0], last_fwd_traj[:, 1], last_fwd_traj[:, 2], 'r--', label=f'Slave (ит. {num_iters})', linewidth=1.8)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.set_title('Траектории на окне WT')
            ax3.legend()
            self.fig_3d.tight_layout()
            self.canvas_3d.draw()

            # Отрисовка Таб 3: График финальной ошибки
            self.fig_diag.clear()
            ax_diag = self.fig_diag.add_subplot(1, 1, 1)
            ax_diag.plot(range(num_iters + 1), iter_errs, 'g-^', markersize=4)
            ax_diag.set_xlabel('Номер итерации')
            ax_diag.set_ylabel('||e(t_0)||')
            ax_diag.set_title('Евклидова норма ошибки в начальной точке t_0')
            ax_diag.grid(True, linestyle='--', alpha=0.5)
            self.fig_diag.tight_layout()
            self.canvas_diag.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = SprottSyncApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
