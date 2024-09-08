import time
import yaml
import socket
import struct
import torch
import numpy as np
from datetime import datetime
from trajopt.l4c_trajectory_generation import ClosedLoopTrajectoryGenerator
from trajopt.rom_dynamics import SingleInt2D
from trajopt.tube_trajopt import problem_dict
import matplotlib.pyplot as plt

def get_send(client_socket, format_str):
    def send_z_v(z, v):
        data = struct.pack(f"{z.size}f{v.size}f", *z.reshape((-1,)).tolist(), *v.reshape((-1,)).tolist())
        # print("Sending z, v: ", z, v)
        client_socket.sendall(data)

    return send_z_v


def get_receive(client_socket):
    def receive_pz_x():
        data = client_socket.recv(16)
        zx, zy, pz_x, pz_y = struct.unpack('4f', data)
        # print("Recieved zx, zy: ", zx, zy, pz_x, pz_y)
        return np.array([zx, zy]), np.array([pz_x, pz_y])

    return receive_pz_x


def main():
    # Read the setup file
    with open('/home/wcompton/Repos/ARCHER_hopper/ControlStack/config/gains.yaml') as f:
        data = yaml.safe_load(f)

    # Initialize Socket Connection
    server_address = ('127.0.0.1', 8081)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)

    # Initialize Connection:
    client_socket.sendall(struct.pack('!?', True))
    N = data["MPC"]["N"]
    send_z_v = get_send(client_socket, "!" + "f" * (N * 4 + 2))
    recieve_pz_x = get_receive(client_socket)

    # Setup problem
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    p_dict = problem_dict[data["MPC"]["prob_str"]]
    p_dict['obs']['cx'] = np.array(data['Simulator']['obsx'])
    p_dict['obs']['cy'] = np.array(data['Simulator']['obsy'])
    p_dict['obs']['r'] = np.array(data['Simulator']['obsr'])
    v_max = data["MPC"]["rom"]["v_max"]
    rom = SingleInt2D(
        data["MPC"]["rom"]["dt"],
        -torch.tensor([np.inf, np.inf], device=device), torch.tensor([np.inf, np.inf], device=device),
        -torch.tensor([v_max, v_max], device=device), torch.tensor([v_max, v_max], device=device),
        n_robots=1,
        backend='torch',
        device=device
    )
    cl_traj_gen = ClosedLoopTrajectoryGenerator(
        rom,
        None,
        N,
        0.1,
        device,
        p_dict,
        'NN_recursive',
        nn_path=data["MPC"]["nn_path"],
        w_max=data["MPC"]["w_max"],
        mpc_dk=data["MPC"]["dk"],
        warm_start='nominal',
        nominal_ws='interpolate',
        track_nominal=True,
        tube_ws="evaluate",
        max_iter=data["MPC"]["max_iter"],
        solver_str="snopt"
    )

    H = 400
    z_k = np.zeros((H + 1, cl_traj_gen.planning_model.n),) * np.nan
    v_k = np.zeros((H, cl_traj_gen.planning_model.m)) * np.nan
    w_k = np.zeros((H + 1, 1)) * np.nan
    pz_x_k = np.zeros_like(z_k) * np.nan

    # mats for visualizing later
    z_vis = np.zeros((H, *z_k.shape))
    v_vis = np.zeros((H, *v_k.shape))
    pz_x_vis = np.zeros((H, *pz_x_k.shape))
    w_vis = np.zeros((H, *w_k.shape))
    z_sol_vis = np.zeros((H, cl_traj_gen.N + 1, cl_traj_gen.planning_model.n))
    v_sol_vis = np.zeros((H, cl_traj_gen.N, cl_traj_gen.planning_model.m))
    w_sol_vis = np.zeros((H, cl_traj_gen.N, 1))
    z_sim = np.zeros((H + 1, cl_traj_gen.planning_model.n),) * np.nan
    cv_vis = {}
    timing = np.zeros((H,))
    terminated = False
    warm_up = True

    t = 0
    t0 = time.perf_counter_ns()
    try:
        while not terminated:
            z0, pz_x0 = recieve_pz_x()
            print("\n\n\n\n\n\ne: ", z0, pz_x0, "\n\n\n\n\n\n\n")
            if warm_up:
                z_k[0, :] = z0
                cl_traj_gen.reset(torch.from_numpy(z0).float(), e_prev=torch.from_numpy(z0 - pz_x0).float())
                warm_up = False
            else:
                cl_traj_gen.step_rom_idx(True, e_prev=torch.from_numpy(z0 - pz_x0).float().to(device),
                                         increment_rom_time=True)
            z_sol, v_sol = cl_traj_gen.trajectory.cpu().numpy(), cl_traj_gen.v_trajectory.cpu().numpy()
            # z_sol, v_sol = cl_traj_gen.nominal_z_warm[None, :, :], cl_traj_gen.nominal_v_warm[None, :, :]
            # plt.plot(cl_traj_gen.nominal_z_warm)
            # plt.xlabel(f'{cl_traj_gen.k.item()}')
            # plt.show()
            # send_z_v(cl_traj_gen.trajectory.cpu().numpy(), cl_traj_gen.v_trajectory.cpu().numpy())
            send_z_v(z_sol, v_sol)

            # Save data
            if t < H:
                z_sim[t, :] = z0
                v_k[t, :] = v_sol[:, 0, :]
                z_k[t + 1, :] = z_sol[:, 1, :]
                pz_x_k[t, :] = pz_x0
                w_k[t + 1] = cl_traj_gen.w_trajectory[0].item()

                z_vis[t] = np.copy(z_k)
                v_vis[t] = np.copy(v_k)
                pz_x_vis[t] = np.copy(pz_x_k)
                w_vis[t] = np.copy(w_k)
                z_sol_vis[t] = np.copy(z_sol)
                v_sol_vis[t] = np.copy(v_sol)
                w_sol_vis[t] = np.copy(cl_traj_gen.w_trajectory)
                cv_vis["cv" + str(t)] = cl_traj_gen.g_dict.copy()
                timing[t] = time.perf_counter_ns() - t0

            t += 1
            if t > H + 2:
                terminated = True
    finally:
        from scipy.io import savemat
        tag = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        fn = f"evaluation/data/{tag}_inner_file.mat"
        cl_traj_gen.write_data(fn)
        print("Saving: ", fn)
        fn = f"evaluation/data/{tag}_outer_file.mat"
        print("Saving: ", fn)
        savemat(fn, {
            "z": z_vis,
            "v": v_vis,
            "w": w_vis,
            "pz_x": pz_x_vis,
            "z_sol": z_sol_vis,
            "v_sol": v_sol_vis,
            "w_sol": w_sol_vis,
            **cv_vis,
            "t": timing,
            "z0": cl_traj_gen.start,
            "zf": cl_traj_gen.goal,
            "obs_x": cl_traj_gen.obs['cx'],
            "obs_y": cl_traj_gen.obs['cy'],
            "obs_r": cl_traj_gen.obs['r'],
            "timing": timing,
            "data": data,
            "z_sim": z_sim
        })

        client_socket.close()


if __name__ == "__main__":
    main()