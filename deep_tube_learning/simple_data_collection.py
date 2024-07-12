import os
import pickle
import random
import string
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from deep_tube_learning.utils import UniformSampleHoldDT
from trajopt.rom_dynamics import SingleInt2D, DoubleInt2D


def main(num_robots, epochs):
    dt = 0.1
    ep_length = 100
    Kp = 1
    Kd = 10

    single_z_max = np.array([np.inf, np.inf])
    single_v_max = np.array([1, 1])
    double_z_max = np.array([np.inf, np.inf, 10, 10])
    double_v_max = np.array([10, 10])
    single_int = SingleInt2D(dt, -single_z_max, single_z_max, -single_v_max, single_v_max, n_robots=num_robots, backend="numpy")
    double_int = DoubleInt2D(dt, -double_z_max, double_z_max, -double_v_max, double_v_max, n_robots=num_robots, backend="numpy")

    run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    data_path = str(Path(__file__).parent / "rom_tracking_data" / f"simple_{run_id}")
    os.makedirs(data_path, exist_ok=True)

    sample_hold_dt = UniformSampleHoldDT(0, 20, n_robots=num_robots)
    for epoch in tqdm(range(epochs)):
        x = np.zeros((int(ep_length) + 1, num_robots, double_int.n))  # Epochs, steps, states
        u = np.zeros((int(ep_length), num_robots, double_int.m))
        z = np.zeros((int(ep_length) + 1, num_robots, single_int.n))
        pz_x = np.zeros((int(ep_length) + 1, num_robots, single_int.n))
        v = np.zeros((int(ep_length), num_robots, single_int.m))
        done = np.zeros((int(ep_length), num_robots), dtype='bool')
        done[-1, :] = False

        # Initialization
        v_nom = single_int.sample_uniform_bounded_v(z[0, :, :])
        t_since_new_v = np.zeros((num_robots,))
        t_sample_hold = sample_hold_dt.sample()

        # Loop over time steps
        for t in range(int(ep_length)):
            # Decide on rom action
            new_v_nom = single_int.sample_uniform_bounded_v(z[t, :, :])
            new_t_sample_hold = sample_hold_dt.sample()

            update_inds = (t_since_new_v >= t_sample_hold)

            v_nom[update_inds, :] = new_v_nom[update_inds, :]
            t_since_new_v[update_inds] = 0
            t_sample_hold[update_inds] = new_t_sample_hold[update_inds]
            t_since_new_v += 1

            vt = single_int.clip_v_z(z[t, :, :], v_nom)

            # Decide fom action
            xt = x[t, :, :]
            zt = z[t, :, :]
            ut = Kp * (zt - xt[:, :2]) + Kd * (vt - xt[:, 2:])

            # Execute rom action
            zt_p1 = single_int.f(z[t, :, :], vt)
            xt_p1 = double_int.f(x[t, :, :], ut)

            # Store
            x[t + 1, :, :] = xt_p1
            u[t, :, :] = ut
            z[t + 1, :, :] = zt_p1
            v[t, :, :] = vt
            pz_x[t + 1, :, :] = double_int.proj_z(xt_p1)

        fig, ax = plt.subplots()
        single_int.plot_spacial(ax, z[:, 0, :], 'k')
        double_int.plot_spacial(ax, x[:, 0, :], 'b')
        plt.legend(['Single', 'Double'])
        plt.axis("square")
        plt.show()

        fig, axs = plt.subplots(2, 1)
        single_int.plot_ts(axs, z[:, 0, :], v[:, 0, :])
        double_int.plot_ts(axs, x[:, 0, :2], x[:-1, 0, 2:])
        plt.show()

        # Log Data
        with open(f"{data_path}/epoch_{epoch}.pickle", "wb") as f:
            epoch_data = {
                'x': x,
                'u': u,
                'z': z,
                'v': v,
                'pz_x': pz_x,
                'done': done
            }
            pickle.dump(epoch_data, f)

    print(f"\nrun ID: {run_id}\ndataset name: simple\nlocal folder: simple_{run_id}")
    return epoch_data


if __name__ == "__main__":
    main(1000, 10)
