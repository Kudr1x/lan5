"""
Модуль численного моделирования и синхронизации хаотической системы Спротта (Sprott Case B).

Методы:
- Симметричный обратимый по времени интегратор CD (Composition Diagonal)
- Однонаправленная синхронизация Пикоры — Кэрола (Pecora-Carroll) по сигналам x, y, z
- Двунаправленная синхронизация (взаимная связь) по сигналам x, y, z
- Быстрая обратимая синхронизация (Time-Reversible Synchronization) на ограниченном временном окне
  по алгоритму Бутусова, Рыбина, Каримова (2025)
"""

import numpy as np
import numba

# ---------------------------------------------------------------------------
# Базовый шаг автономной системы Sprott B методом CD
# dx/dt = y * z
# dy/dt = x - y
# dz/dt = 1 - x * y
# ---------------------------------------------------------------------------

@numba.njit(fastmath=True)
def sprott_step_cd(x, y, z, h):
    """
    Один шаг автономного интегратора CD второго порядка точности.
    Композиция полуявного метода Эйлера и сопряженного полунеявного метода.
    Обладает строгой обратимостью по времени (точность обращения до эпсилон машины).
    """
    tau = 0.5 * h
    # Подшаг 1: порядок обхода x -> y -> z
    x1 = x + tau * (y * z)
    y1 = y + tau * (x1 - y)
    z1 = z + tau * (1.0 - x1 * y1)

    # Подшаг 2 (сопряженный): порядок обхода z -> y -> x
    z2 = z1 + tau * (1.0 - x1 * y1)
    y2 = (y1 + tau * x1) / (1.0 + tau)
    x2 = x1 + tau * (y2 * z2)

    return x2, y2, z2


@numba.njit(fastmath=True)
def sprott_step_coupled(x, y, z, xm, ym, zm, h, K, var_idx):
    """
    Шаг CD с добавлением контроллера диффузной связи Пикоры-Кэрола:
    var_idx: 0 -> связь по x, 1 -> связь по y, 2 -> связь по z
    Контроллер добавляется к производной соответствующей переменной:
    C_u = K * (u_m - u)
    """
    tau = 0.5 * h
    cx = K * (xm - x) if var_idx == 0 else 0.0
    cy = K * (ym - y) if var_idx == 1 else 0.0
    cz = K * (zm - z) if var_idx == 2 else 0.0

    # Подшаг 1
    x1 = x + tau * (y * z + cx)
    y1 = y + tau * (x1 - y + cy)
    z1 = z + tau * (1.0 - x1 * y1 + cz)

    # Подшаг 2
    z2 = z1 + tau * (1.0 - x1 * y1 + cz)
    y2 = (y1 + tau * (x1 + cy)) / (1.0 + tau)
    x2 = x1 + tau * (y2 * z2 + cx)

    return x2, y2, z2


@numba.njit(fastmath=True)
def sprott_step_bidir(xm, ym, zm, xs, ys, zs, h, K, var_idx):
    """
    Шаг взаимной (двунаправленной) связи:
    Мастер получает контроллер: +K * (u_s - u_m)
    Слейв получает контроллер:  +K * (u_m - u_s)
    """
    tau = 0.5 * h
    cx_m = K * (xs - xm) if var_idx == 0 else 0.0
    cy_m = K * (ys - ym) if var_idx == 1 else 0.0
    cz_m = K * (zs - zm) if var_idx == 2 else 0.0

    cx_s = -cx_m
    cy_s = -cy_m
    cz_s = -cz_m

    # Мастер: Подшаг 1
    xm1 = xm + tau * (ym * zm + cx_m)
    ym1 = ym + tau * (xm1 - ym + cy_m)
    zm1 = zm + tau * (1.0 - xm1 * ym1 + cz_m)
    # Мастер: Подшаг 2
    zm2 = zm1 + tau * (1.0 - xm1 * ym1 + cz_m)
    ym2 = (ym1 + tau * (xm1 + cy_m)) / (1.0 + tau)
    xm2 = xm1 + tau * (ym2 * zm2 + cx_m)

    # Слейв: Подшаг 1
    xs1 = xs + tau * (ys * zs + cx_s)
    ys1 = ys + tau * (xs1 - ys + cy_s)
    zs1 = zs + tau * (1.0 - xs1 * ys1 + cz_s)
    # Слейв: Подшаг 2
    zs2 = zs1 + tau * (1.0 - xs1 * ys1 + cz_s)
    ys2 = (ys1 + tau * (xs1 + cy_s)) / (1.0 + tau)
    xs2 = xs1 + tau * (ys2 * zs2 + cx_s)

    return xm2, ym2, zm2, xs2, ys2, zs2


# ---------------------------------------------------------------------------
# Генерация траекторий и симуляция
# ---------------------------------------------------------------------------

@numba.njit(fastmath=True)
def simulate_unidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps, K, var_idx):
    """
    Моделирование однонаправленной синхронизации.
    Возвращает массивы траекторий мастера, слейва и нормы ошибки.
    """
    traj_m = np.zeros((steps, 3))
    traj_s = np.zeros((steps, 3))
    errors = np.zeros(steps)

    xm, ym, zm = x0_m, y0_m, z0_m
    xs, ys, zs = x0_s, y0_s, z0_s

    for i in range(steps):
        traj_m[i, 0] = xm
        traj_m[i, 1] = ym
        traj_m[i, 2] = zm

        traj_s[i, 0] = xs
        traj_s[i, 1] = ys
        traj_s[i, 2] = zs

        dx = xm - xs
        dy = ym - ys
        dz = zm - zs
        errors[i] = np.sqrt(dx * dx + dy * dy + dz * dz)

        # Вычисление связи на текущем шаге
        cx = K * dx if var_idx == 0 else 0.0
        cy = K * dy if var_idx == 1 else 0.0
        cz = K * dz if var_idx == 2 else 0.0

        # Шаг мастера
        xm, ym, zm = sprott_step_cd(xm, ym, zm, h)

        # Шаг слейва
        tau = 0.5 * h
        xs1 = xs + tau * (ys * zs + cx)
        ys1 = ys + tau * (xs1 - ys + cy)
        zs1 = zs + tau * (1.0 - xs1 * ys1 + cz)

        zs2 = zs1 + tau * (1.0 - xs1 * ys1 + cz)
        ys2 = (ys1 + tau * (xs1 + cy)) / (1.0 + tau)
        xs2 = xs1 + tau * (ys2 * zs2 + cx)
        xs, ys, zs = xs2, ys2, zs2

    return traj_m, traj_s, errors


@numba.njit(fastmath=True)
def simulate_bidir(x0_m, y0_m, z0_m, x0_s, y0_s, z0_s, h, steps, K, var_idx):
    """
    Моделирование двунаправленной (взаимной) синхронизации.
    """
    traj_m = np.zeros((steps, 3))
    traj_s = np.zeros((steps, 3))
    errors = np.zeros(steps)

    xm, ym, zm = x0_m, y0_m, z0_m
    xs, ys, zs = x0_s, y0_s, z0_s

    for i in range(steps):
        traj_m[i, 0] = xm
        traj_m[i, 1] = ym
        traj_m[i, 2] = zm

        traj_s[i, 0] = xs
        traj_s[i, 1] = ys
        traj_s[i, 2] = zs

        dx = xm - xs
        dy = ym - ys
        dz = zm - zs
        errors[i] = np.sqrt(dx * dx + dy * dy + dz * dz)

        xm, ym, zm, xs, ys, zs = sprott_step_bidir(xm, ym, zm, xs, ys, zs, h, K, var_idx)

    return traj_m, traj_s, errors


@numba.njit(fastmath=True)
def run_reversible_sync(master_window, x0_s, y0_s, z0_s, h, K_plus, K_minus, var_idx, num_iters):
    """
    Алгоритм обратимой синхронизации (Time-Reversible Synchronization).
    master_window: массив формы (N+1, 3) значений траектории ведущей системы на окне WT
    x0_s, y0_s, z0_s: начальное состояние ведомой системы при t=0
    K_plus: коэффициент связи для прямого хода
    K_minus: коэффициент связи для обратного хода
    num_iters: число циклов (прямой + обратный ход)

    Возвращает:
    - iter_errors: массив евклидовой ошибки в начале окна t=0 на каждой итерации (длина num_iters+1)
    - final_xs, final_ys, final_zs: состояние слейва при t=0 после всех итераций
    - last_forward_traj: траектория прямого хода слейва на последней итерации
    """
    N_steps = master_window.shape[0] - 1
    iter_errors = np.zeros(num_iters + 1)

    xs, ys, zs = x0_s, y0_s, z0_s
    e0 = np.sqrt((master_window[0, 0] - xs)**2 +
                 (master_window[0, 1] - ys)**2 +
                 (master_window[0, 2] - zs)**2)
    iter_errors[0] = e0

    last_fwd_traj = np.zeros((N_steps + 1, 3))

    for it in range(num_iters):
        # 1. Прямой ход (от n=0 до N_steps, h > 0)
        for step in range(N_steps):
            if it == num_iters - 1:
                last_fwd_traj[step, 0] = xs
                last_fwd_traj[step, 1] = ys
                last_fwd_traj[step, 2] = zs

            xm_n = master_window[step, 0]
            ym_n = master_window[step, 1]
            zm_n = master_window[step, 2]
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, xm_n, ym_n, zm_n, h, K_plus, var_idx)

        if it == num_iters - 1:
            last_fwd_traj[N_steps, 0] = xs
            last_fwd_traj[N_steps, 1] = ys
            last_fwd_traj[N_steps, 2] = zs

        # 2. Обратный ход (от n=N_steps вниз до 1, шаг -h, коэффициент -K_minus)
        for step in range(N_steps, 0, -1):
            xm_n = master_window[step, 0]
            ym_n = master_window[step, 1]
            zm_n = master_window[step, 2]
            xs, ys, zs = sprott_step_coupled(xs, ys, zs, xm_n, ym_n, zm_n, -h, -K_minus, var_idx)

        # Ошибка в начальной точке t=0 после завершения цикла
        err_t0 = np.sqrt((master_window[0, 0] - xs)**2 +
                         (master_window[0, 1] - ys)**2 +
                         (master_window[0, 2] - zs)**2)
        iter_errors[it + 1] = err_t0

    return iter_errors, xs, ys, zs, last_fwd_traj
