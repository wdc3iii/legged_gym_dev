import numpy as np
import matplotlib.pyplot as plt

from trajopt.rom_dynamics import (SingleInt2D, DoubleInt2D, Unicycle, LateralUnicycle,
                                  ExtendedUnicycle, ExtendedLateralUnicycle)

dt = 0.1
N = 10
E = 10
acc_max = 2
alpha_max = 4
vel_max = 1
omega_max = 2
pos_max = np.inf


def test_numpy_rom(pm):
    z0 = np.zeros((pm.n,))
    zt = np.zeros((N * E + 1, pm.n))
    vt = np.zeros((N * E, pm.m))
    zt[0, :] = z0

    for e in range(E):
        v = pm.sample_uniform_bounded_v(zt[e * N, :])
        for k in range(N):
            t = e * N + k
            v_clip = pm.clip_v(zt[t, :], v)
            vt[t, :] = v_clip
            zt[t + 1, :] = pm.f(zt[t, :], v_clip)

    fig, axs = plt.subplots(2, 1)
    pm.plot_ts(axs, zt, vt)
    plt.show()

    fig, ax = plt.subplots()
    pm.plot_spacial(ax, zt)
    plt.axis("square")
    plt.show()


def test_numpy_single_int():
    z_max = np.array([pos_max, pos_max])
    v_max = np.array([vel_max, vel_max])
    pm = SingleInt2D(dt, -z_max, z_max, -v_max, v_max, backend="numpy")
    test_numpy_rom(pm)


def test_numpy_double_int():
    z_max = np.array([pos_max, pos_max, vel_max, vel_max])
    v_max = np.array([acc_max, acc_max])
    pm = DoubleInt2D(dt, -z_max, z_max, -v_max, v_max, backend="numpy")
    test_numpy_rom(pm)


def test_numpy_unicycle():
    z_max = np.array([pos_max, pos_max, np.inf])
    v_max = np.array([vel_max, omega_max])
    v_min = -np.array([vel_max / 2, omega_max])
    pm = Unicycle(dt, -z_max, z_max, v_min, v_max, backend="numpy")
    test_numpy_rom(pm)


def test_numpy_lateral_unicycle():
    z_max = np.array([pos_max, pos_max, np.inf])
    v_max = np.array([vel_max, vel_max / 2, omega_max])
    v_min = np.array([-vel_max / 2, -vel_max / 2, -omega_max])
    pm = LateralUnicycle(dt, -z_max, z_max, v_min, v_max, backend="numpy")
    test_numpy_rom(pm)


def test_numpy_extended_unicycle():
    z_max = np.array([pos_max, pos_max, np.inf, vel_max, omega_max])
    z_min = -np.array([pos_max, pos_max, np.inf, vel_max / 2, omega_max])
    v_max = np.array([acc_max, alpha_max])
    pm = ExtendedUnicycle(dt, z_min, z_max, -v_max, v_max, backend="numpy")
    test_numpy_rom(pm)


def test_numpy_extended_lateral_unicycle():
    z_max = np.array([pos_max, pos_max, np.inf, vel_max, vel_max / 2, omega_max])
    z_min = -np.array([pos_max, pos_max, np.inf, vel_max / 2, vel_max / 2, omega_max])
    v_max = np.array([acc_max, acc_max / 2, alpha_max])
    pm = ExtendedLateralUnicycle(dt, z_min, z_max, -v_max, v_max, backend="numpy")
    test_numpy_rom(pm)


if __name__ == "__main__":
    # test_numpy_single_int()
    # test_numpy_double_int()
    # test_numpy_unicycle()
    # test_numpy_lateral_unicycle()
    # test_numpy_extended_unicycle()
    test_numpy_extended_lateral_unicycle()
