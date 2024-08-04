import numpy as np
import matplotlib.pyplot as plt
from deep_tube_learning.utils import UniformSampleHoldDT, WeightSamplerSampleAndHold

from trajopt.rom_dynamics import (SingleInt2D, DoubleInt2D, Unicycle, LateralUnicycle,
                                  ExtendedUnicycle, ExtendedLateralUnicycle, TrajectoryGenerator,
                                  ZeroTrajectoryGenerator, SquareTrajectoryGenerator, CircleTrajectoryGenerator)
import torch


dt = 0.02
N = 500
acc_max = 1.0
alpha_max = 4.0
vel_max = 1.0
omega_max = 2.0
pos_max = np.inf

num_robots = 3


def test_numpy_rom(pm):
    t_samp = UniformSampleHoldDT(0.1, 2)
    w_samp = WeightSamplerSampleAndHold()
    traj_gen = TrajectoryGenerator(pm, t_samp, w_samp, prob_stationary=.0001)
    # traj_gen.reset()

    z0 = np.zeros((num_robots, pm.n,))
    zt = np.zeros((N + 1, num_robots, pm.n))
    vt = np.zeros((N, num_robots, pm.m))
    zt[0, :] = z0

    t_swap = []
    t_final_prev = traj_gen.t_final[0]
    for t in range(N):
        v_clip = traj_gen.get_input_t(t * pm.dt, zt[t, :, :])
        if t > 1 and traj_gen.t_final[0] != t_final_prev:
            t_final_prev = traj_gen.t_final[0]
            t_swap.append(t)
        vt[t, :] = v_clip
        zt[t + 1, :] = pm.f(zt[t, :, :], v_clip)

    fig, axs = plt.subplots(2, 1)
    pm.plot_ts(axs, zt[:, 0, :], vt[:, 0, :])
    plt.vlines(np.array(t_swap) * pm.dt, np.min(pm.v_min), np.max(pm.v_max), 'r')
    plt.show()

    fig, ax = plt.subplots()
    pm.plot_spacial(ax, zt[:, 0, :])
    plt.axis("square")
    plt.show()


def test_numpy_single_int():
    z_max = np.array([pos_max, pos_max])
    v_max = np.array([vel_max, vel_max])
    pm = SingleInt2D(dt, -z_max, z_max, -v_max, v_max, n_robots=num_robots, backend="numpy")
    test_numpy_rom(pm)


def test_numpy_double_int():
    z_max = np.array([pos_max, pos_max, vel_max, vel_max])
    v_max = np.array([acc_max, acc_max])
    pm = DoubleInt2D(dt, -z_max, z_max, -v_max, v_max, n_robots=num_robots, backend="numpy")
    test_numpy_rom(pm)


def test_numpy_unicycle():
    z_max = np.array([pos_max, pos_max, np.inf])
    v_max = np.array([vel_max, omega_max])
    v_min = -np.array([vel_max / 2, omega_max])
    pm = Unicycle(dt, -z_max, z_max, v_min, v_max, n_robots=num_robots, backend="numpy")
    test_numpy_rom(pm)


def test_numpy_lateral_unicycle():
    z_max = np.array([pos_max, pos_max, np.inf])
    v_max = np.array([vel_max, vel_max / 2, omega_max])
    v_min = np.array([-vel_max / 2, -vel_max / 2, -omega_max])
    pm = LateralUnicycle(dt, -z_max, z_max, v_min, v_max, n_robots=num_robots, backend="numpy")
    test_numpy_rom(pm)


def test_numpy_extended_unicycle():
    z_max = np.array([pos_max, pos_max, np.inf, vel_max, omega_max])
    z_min = -np.array([pos_max, pos_max, np.inf, vel_max / 2, omega_max])
    v_max = np.array([acc_max, alpha_max])
    pm = ExtendedUnicycle(dt, z_min, z_max, -v_max, v_max, n_robots=num_robots, backend="numpy")
    test_numpy_rom(pm)


def test_numpy_extended_lateral_unicycle():
    z_max = np.array([pos_max, pos_max, np.inf, vel_max, vel_max / 2, omega_max])
    z_min = -np.array([pos_max, pos_max, np.inf, vel_max / 2, vel_max / 2, omega_max])
    v_max = np.array([acc_max, acc_max / 2, alpha_max])
    pm = ExtendedLateralUnicycle(dt, z_min, z_max, -v_max, v_max, n_robots=num_robots, backend="numpy")
    test_numpy_rom(pm)


def test_torch_single_stationary():
    z_max = torch.tensor([pos_max, pos_max], device='cuda')
    v_max = torch.tensor([vel_max, vel_max], device='cuda')
    pm = SingleInt2D(dt, -z_max, z_max, -v_max, v_max, n_robots=num_robots, backend="torch")
    traj_gen = ZeroTrajectoryGenerator(pm, None, None, N=500, backend="torch")

    z0 = torch.zeros((num_robots, pm.n,), device='cuda')
    traj_gen.reset(z0)
    zt = traj_gen.trajectory.cpu().numpy()

    rbot = 1
    fig, ax = plt.subplots()
    pm.plot_spacial(ax, zt[rbot, :, :])
    plt.axis("square")
    plt.show()

def test_torch_double_stationary():
    z_max = torch.tensor([pos_max, pos_max, vel_max, vel_max], device='cuda')
    v_max = torch.tensor([acc_max, acc_max], device='cuda')
    pm = DoubleInt2D(dt, -z_max, z_max, -v_max, v_max, n_robots=num_robots, backend="torch")
    traj_gen = ZeroTrajectoryGenerator(pm, None, None, N=500, backend="torch")

    z0 = torch.zeros((num_robots, pm.n,), device='cuda')
    traj_gen.reset(z0)
    zt = traj_gen.trajectory.cpu().numpy()

    rbot = 1
    fig, ax = plt.subplots()
    pm.plot_spacial(ax, zt[rbot, :, :])
    plt.axis("square")
    plt.show()

def test_torch_double_square():
    z_max = torch.tensor([pos_max, pos_max, vel_max, vel_max], device='cuda')
    v_max = torch.tensor([acc_max, acc_max], device='cuda')
    pm = DoubleInt2D(dt, -z_max, z_max, -v_max, v_max, n_robots=num_robots, backend="torch")

    traj_gen = SquareTrajectoryGenerator(pm, None, None, N=500, backend="torch")

    z0 = torch.zeros((num_robots, pm.n,), device='cuda')
    traj_gen.reset(z0)
    zt = traj_gen.trajectory.cpu().numpy()

    rbot = 1
    fig, ax = plt.subplots()
    pm.plot_spacial(ax, zt[rbot, :, :])
    plt.axis("square")
    plt.show()

def test_torch_single_square():
    z_max = torch.tensor([pos_max, pos_max], device='cuda')
    v_max = torch.tensor([vel_max, vel_max], device='cuda')
    pm = SingleInt2D(dt, -z_max, z_max, -v_max, v_max, n_robots=num_robots, backend="torch")
    traj_gen = SquareTrajectoryGenerator(pm, None, None, N=500, backend="torch")

    z0 = torch.zeros((num_robots, pm.n,), device='cuda')
    traj_gen.reset(z0)
    zt = traj_gen.trajectory.cpu().numpy()

    rbot = 1
    fig, ax = plt.subplots()
    pm.plot_spacial(ax, zt[rbot, :, :])
    plt.axis("square")
    plt.show()


def test_torch_double_circle():
    z_max = torch.tensor([pos_max, pos_max, vel_max, vel_max], device='cuda')
    v_max = torch.tensor([acc_max, acc_max], device='cuda')
    pm = DoubleInt2D(dt, -z_max, z_max, -v_max, v_max, n_robots=num_robots, backend="torch")

    traj_gen = CircleTrajectoryGenerator(pm, None, None, N=1000, backend="torch")

    z0 = torch.zeros((num_robots, pm.n,), device='cuda')
    traj_gen.reset(z0)
    zt = traj_gen.trajectory.cpu().numpy()

    rbot = 1
    fig, ax = plt.subplots()
    pm.plot_spacial(ax, zt[rbot, :, :])
    plt.axis("square")
    plt.show()

def test_torch_single_circle():
    z_max = torch.tensor([pos_max, pos_max], device='cuda')
    v_max = torch.tensor([vel_max, vel_max], device='cuda')
    pm = SingleInt2D(dt, -z_max, z_max, -v_max, v_max, n_robots=num_robots, backend="torch")
    traj_gen = CircleTrajectoryGenerator(pm, None, None, N=500, backend="torch")

    z0 = torch.zeros((num_robots, pm.n,), device='cuda')
    traj_gen.reset(z0)
    zt = traj_gen.trajectory.cpu().numpy()

    rbot = 1
    fig, ax = plt.subplots()
    pm.plot_spacial(ax, zt[rbot, :, :])
    plt.axis("square")
    plt.show()


if __name__ == "__main__":
    # test_numpy_single_int()
    # test_numpy_double_int()
    # test_numpy_unicycle()
    # test_numpy_lateral_unicycle()
    # test_numpy_extended_unicycle()
    # test_numpy_extended_lateral_unicycle()
    test_torch_single_stationary()
    test_torch_double_stationary()
    test_torch_single_square()
    test_torch_double_square()
    test_torch_single_circle()
    test_torch_double_circle()