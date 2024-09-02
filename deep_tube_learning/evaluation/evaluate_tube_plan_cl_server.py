from trajopt.tube_trajopt import *
import pickle
from omegaconf import OmegaConf
from deep_tube_learning.utils import unnormalize_dict
import time
import torch
from deep_tube_learning.custom_sim import CustomSim
import socket
import struct


track_warm = True

warm_start = 'nominal'

tube_ws = "evaluate"

# tube_dyn = "NN_oneshot"
tube_dyn = "NN_recursive"
# nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/rkm53z6t"  # N = 50
# nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/nqkkk3af"  # N = 10
nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/y87jn2r7"  # H2H

max_iter = 200
N = 10
mpc_dk = 1
Rv1 = 10
Rv2 = 10

def arr2list(d):
    if type(d) is dict:
        return {key: arr2list(val) for key, val in d.items()}
    elif type(d) is np.ndarray:
        return [arr2list(v) for v in list(d)]
    elif type(d) is np.float64:
        return float(d)
    elif type(d) is np.int64:
        return int(d)
    else:
        return d


def get_send(server_socket, client_address):
    def send_z_v(z, v, done):
        data = struct.pack('!Bffff', done, *z.squeeze().tolist(), *v.squeeze().tolist())
        # print("Sending z, v, d: ", z, v, done)
        server_socket.sendto(data, client_address)
    return send_z_v


def get_receive(server_socket, device):
    def receive_pz_x():
        data, _ = server_socket.recvfrom(8)
        zx, zy = struct.unpack('!ff', data)
        # print("recieved zx, zy: ", zx, zy)
        return torch.tensor([[zx, zy]], device=device)
    return receive_pz_x


def main():
    # Socket setup, to talk to IsaacGym in different python env
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('127.0.0.1', 12345))
    prob_str, client_socket_address = server_socket.recvfrom(1024)  # Buffer size is 1024 bytes
    prob_str = prob_str.decode()
    H, socket_address = server_socket.recvfrom(4)  # Buffer size is 4 bytes
    H = struct.unpack('!I', H)[0]
    print("Prob_str: ", prob_str)
    print("H: ", H)
    send_z_v = get_send(server_socket, client_socket_address)

    server_socket.sendto(nn_path.encode(), socket_address)

    if Rv1 is not None:
        problem_dict[prob_str]['Rv_first'] = Rv1
    if Rv2 is not None:
        problem_dict[prob_str]['Rv_second'] = Rv2

    model_name = f'{nn_path}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)

    run_id = model_cfg.dataset.wandb_experiment
    with open(f"../rom_tracking_data/{run_id}/config.pickle", 'rb') as f:
        dataset_cfg = pickle.load(f)
    dataset_cfg = unnormalize_dict(dataset_cfg)
    dataset_cfg['env_config']['env']['num_envs'] = 1
    dataset_cfg['env_config']['env']['type'] = 'custom'
    dataset_cfg['env_config']['env']['model'] = {
        'dt': 0.02,
        'cls': 'ZeroInt2D',
        'z_min': -1e9,
        'z_max': 1e9,
        'v_min': -1e9,
        'v_max': 1e9
    }
    dataset_cfg['env_config']['trajectory_generator'] = {
        'cls': 'ClosedLoopTrajectoryGenerator',
        'H': H,
        'N': N,
        'dt_loop': dataset_cfg['env_config']['env']['model']['dt'],
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'prob_dict': {key: arr2list(val) for key, val in problem_dict[prob_str].items()},
        'tube_dyn': tube_dyn,
        'nn_path': nn_path,
        'w_max': 1,
        'mpc_dk': mpc_dk,
        'warm_start': warm_start,
        'nominal_ws': 'interpolate',
        'track_nominal': track_warm,
        'tube_ws': tube_ws,
        'max_iter': max_iter
    }
    # dataset_cfg['env_config']['rom']['v_max'] = [0.15, 0.15]
    # dataset_cfg['env_config']['rom']['v_min'] = [-0.15, -0.15]
    dataset_cfg['env_config']['domain_rand']['randomize_rom_distance'] = False
    dataset_cfg['env_config']['domain_rand']['zero_rom_dist_llh'] = 1.0
    dataset_cfg['env_config']['init_state']['default_noise_lower'] = [0.0, 0.0]
    dataset_cfg['env_config']['init_state']['default_noise_upper'] = [0.0, 0.0]
    dataset_cfg = OmegaConf.create(unnormalize_dict(dataset_cfg))

    # Define a new custom sim
    env = CustomSim(dataset_cfg.env_config)
    receive_pz_x = get_receive(server_socket, env.device)

    tube_ws_str = str(tube_ws).replace('.', '_')
    z_k = torch.zeros((H + 1, env.traj_gen.planning_model.n), device=env.device) * torch.nan
    v_k = torch.zeros((H, env.traj_gen.planning_model.m), device=env.device) * torch.nan
    x = torch.zeros((1, z_k.shape[0], env.model.n), device=env.device) * torch.nan
    w_k = torch.zeros((H + 1, 1), device=env.device) * torch.nan
    pz_x_k = torch.zeros_like(z_k, device=env.device) * torch.nan

    # mats for visualizing later
    z_vis = torch.zeros((H, *z_k.shape), device=env.device)
    v_vis = torch.zeros((H, *v_k.shape), device=env.device)
    pz_x_vis = torch.zeros((H, *pz_x_k.shape), device=env.device)
    w_vis = torch.zeros((H, *w_k.shape), device=env.device)
    z_sol_vis = torch.zeros((H, env.traj_gen.N + 1, env.traj_gen.planning_model.n), device=env.device)
    v_sol_vis = torch.zeros((H, env.traj_gen.N, env.traj_gen.planning_model.m), device=env.device)
    w_sol_vis = torch.zeros((H, env.traj_gen.N, 1), device=env.device)
    cv_vis = {}
    timing = np.zeros((H,))

    env.reset()
    z_k[0, :] = env.traj_gen.get_trajectory()[:, 0, :]
    x[:, 0, :] = env.get_states()
    pz_x_k[0, :] = env.model.proj_z(x[:, 0, :])
    w_k[0] = torch.linalg.norm(z_k[0, :] - pz_x_k[0, :])

    t0 = time.perf_counter_ns()

    for t in range(H):
        # Step environment until rom steps
        k = torch.clone(env.traj_gen.k.detach())
        while torch.any(env.traj_gen.k == k):
            pz_x = receive_pz_x()
            obs, _, _, dones, _ = env.step(pz_x)
            send_z_v(env.traj_gen.get_trajectory()[:, dataset_cfg.controller.N, :], torch.clone(env.traj_gen.get_v_trajectory()[:, dataset_cfg.controller.N, :].detach()), False)

        # Save Data
        base = torch.clone(env.root_states.detach())
        d = torch.clone(dones.detach())
        proj = env.rom.proj_z(base)
        v_k[t, :] = env.traj_gen.v_trajectory[:, 0, :]
        x[:, t + 1, :] = env.get_states()
        z_k[t + 1, :] = env.traj_gen.get_trajectory()[:, 0, :]
        pz_x_k[t + 1, :] = proj
        w_k[t + 1] = env.traj_gen.w_trajectory[0].item()

        z_vis[t] = torch.clone(z_k)
        v_vis[t] = torch.clone(v_k)
        pz_x_vis[t] = torch.clone(pz_x_k)
        w_vis[t] = torch.clone(w_k)
        z_sol_vis[t] = torch.clone(env.traj_gen.trajectory)
        v_sol_vis[t] = torch.clone(env.traj_gen.v_trajectory)
        w_sol_vis[t] = torch.clone(torch.from_numpy(env.traj_gen.w_trajectory).float()).to(env.device)
        cv_vis["cv" + str(t)] = env.traj_gen.g_dict.copy()
        timing[t] = time.perf_counter_ns() - t0

    send_z_v(torch.tensor([[0., 0.]]), torch.tensor([[0., 0.]]), True)
    print("Closing Server Socket...")
    server_socket.close()

    from scipy.io import savemat
    fn = f"data/cl_tube_{prob_str}_{nn_path[-8:]}_{warm_start}_Rv_{Rv1}_{Rv2}_N_{N}_dk_{mpc_dk}_{tube_dyn}_{tube_ws_str}_{track_warm}.mat"
    savemat(fn, {
        "z": z_vis.detach().cpu().numpy(),
        "v": v_vis.detach().cpu().numpy(),
        "w": w_vis.detach().cpu().numpy(),
        "pz_x": pz_x_vis.detach().cpu().numpy(),
        "z_sol": z_sol_vis.detach().cpu().numpy(),
        "v_sol": v_sol_vis.detach().cpu().numpy(),
        "w_sol": w_sol_vis.detach().cpu().numpy(),
        **cv_vis,
        "t": timing,
        "z0": env.traj_gen.start,
        "zf": env.traj_gen.goal,
        "obs_x": env.traj_gen.obs['cx'],
        "obs_y": env.traj_gen.obs['cy'],
        "obs_r": env.traj_gen.obs['r'],
        "timing": timing
    })

    print(f"Complete! Writing to {fn}")
    print(f"Time Solving Nominal: {env.traj_gen.t_solving_nominal:.4f} \tRate: {env.traj_gen.t_solving_nominal / H:.4f}")
    print(f"Time Solving Tube:    {env.traj_gen.t_solving:.4f} \tRate: {env.traj_gen.t_solving / H:.4f}")


if __name__ == '__main__':
    main()
