import os
import pickle
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from deep_tube_learning.utils import UniformWeightSamplerNoRamp, UniformSampleHoldDT, torch_rand_vec_float
from trajopt.rom_dynamics import SingleInt2D, DoubleInt2D, TrajectoryGenerator
import torch


def main(num_robots, epochs, max_rom_dist=0.5, zero_err_prob=0.25):
    dt = 0.1
    ep_length = 100
    Kp = 10
    Kd = 10
    save_debugging_data = False
    upload_to_wandb = False
    max_rom_distance = torch.tensor([max_rom_dist, max_rom_dist])

    device = torch.device('cpu')

    single_z_max = torch.tensor([np.inf, np.inf])
    single_v_max = torch.tensor([0.2, 0.2])
    double_z_max = torch.tensor([np.inf, np.inf, 0.3, 0.3])
    double_v_max = torch.tensor([0.5, 0.5])
    single_int = SingleInt2D(dt, -single_z_max, single_z_max, -single_v_max, single_v_max, n_robots=num_robots, device=device, backend='torch')
    double_int = DoubleInt2D(dt, -double_z_max, double_z_max, -double_v_max, double_v_max, n_robots=num_robots, device=device, backend='torch')

    # Send config to wandb
    cfg_dict = pd.json_normalize({
        "dt": dt, "ep_length": 100, "Kp": 10, "Kd": 10, "save_debugging_data": save_debugging_data, "upload_to_wandb": upload_to_wandb,
        "v_max": list(single_v_max), "z_max": list(single_z_max)

    }, sep="/").to_dict(orient="records")[0]
    if upload_to_wandb:
        wandb.init(project="RoM_Tracking_Data",
                   entity="coleonguard-Georgia Institute of Technology",
                   name="simple",  # Use the dynamic experiment name
                   config=cfg_dict)
        run_id = wandb.run.id
    else:
        import random
        import string
        run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    data_path = str(Path(__file__).parent / "rom_tracking_data" / f"simple_{run_id}")
    os.makedirs(data_path, exist_ok=True)

    traj_gen = TrajectoryGenerator(
        t_sampler=UniformSampleHoldDT(0.01, 2, backend='torch', device='cpu'),
        weight_sampler=UniformWeightSamplerNoRamp(device='cpu'),
        rom=single_int,
        backend='torch',
        device=device,
        dt_loop=0.1
    )

    for epoch in tqdm(range(epochs)):
        x = torch.zeros((num_robots, int(ep_length) + 1, double_int.n))  # Epochs, steps, states
        u = torch.zeros((num_robots, int(ep_length), double_int.m))
        z = torch.zeros((num_robots, int(ep_length) + 1, single_int.n))
        pz_x = torch.zeros((num_robots, int(ep_length) + 1, single_int.n))
        v = torch.zeros((num_robots, int(ep_length), single_int.m))
        done = torch.zeros((num_robots, int(ep_length)), dtype=torch.bool)

        # Initialization
        z0 = z[:, 0, :]
        mask = torch.rand(num_robots) > zero_err_prob
        z0[mask, :] += torch_rand_vec_float(-max_rom_distance, max_rom_distance, z0[mask, :].shape, device)

        traj_gen.reset(z0)

        # Loop over time steps
        for t in range(int(ep_length)):
            # Decide fom action
            xt = x[:, t, :]
            zt = z[:, t, :]
            # zt_p = traj_gen.trajectory[:, 1, :]
            vt = traj_gen.v_trajectory[:, 0, :]
            vt_p = traj_gen.v_trajectory[:, 1, :]
            ut = double_int.clip_v_z(xt, Kp * (zt - xt[:, :2]) + Kd * (vt_p - xt[:, 2:]))

            xt_p1 = double_int.f(xt, ut)

            # Store
            x[:, t + 1, :] = xt_p1
            u[:, t, :] = ut
            z[:, t + 1, :] = traj_gen.get_trajectory()[:, 1, :]
            v[:, t, :] = vt
            pz_x[:, t + 1, :] = double_int.proj_z(xt_p1)
            traj_gen.step()

        fig, ax = plt.subplots()
        single_int.plot_spacial(ax, z[0, :, :], 'k')
        double_int.plot_spacial(ax, x[0, :, :])
        plt.legend(['Single', 'Double'])
        plt.axis("square")
        plt.show()

        # fig, axs = plt.subplots(2, 1)
        # single_int.plot_ts(axs, z[0, :, :], v[0, :, :])
        # single_int.plot_ts(axs, x[0, :, :2], x[0, :-1, 2:])
        # plt.show()
        #
        # fig, axs = plt.subplots(2, 1)
        # double_int.plot_ts(axs, x[0, :, :], u[0, :, :])
        # plt.show()

        # Log Data
        with open(f"{data_path}/epoch_{epoch}.pickle", "wb") as f:
            if save_debugging_data:
                epoch_data = {
                    'x': x.cpu().detach().numpy(),
                    'u': u.cpu().detach().numpy(),
                    'z': z.cpu().detach().numpy(),
                    'v': v.cpu().detach().numpy(),
                    'pz_x': pz_x.cpu().detach().numpy(),
                    'done': done.cpu().detach().numpy()
                }
            else:
                epoch_data = {
                    'z': z.cpu().detach().numpy(),
                    'v': v.cpu().detach().numpy(),
                    'pz_x': pz_x.cpu().detach().numpy(),
                    'done': done.cpu().detach().numpy()
                }
            pickle.dump(epoch_data, f)

    print(f"\nrun ID: {run_id}\ndataset name: simple\nlocal folder: simple_{run_id}")
    return epoch_data


if __name__ == "__main__":
    main(8192, 100)
