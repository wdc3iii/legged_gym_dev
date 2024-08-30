from trajopt.tube_trajopt import *
import pickle
from omegaconf import OmegaConf
from deep_tube_learning.utils import unnormalize_dict
import time
import torch
from deep_tube_learning.custom_sim import CustomSim


# prob_str = 'right'
# prob_str = 'right_wide'
# prob_str = 'gap'
prob_str = 'gap_big'

track_warm = True

# warm_start = 'start'
# warm_start = 'goal'
# warm_start = 'interpolate'
warm_start = 'nominal'

# tube_ws = 0
# tube_ws = 0.5
tube_ws = "evaluate"

# tube_dyn = 'l1'
# tube_dyn = "l2"
# tube_dyn = "l1_rolling"
# tube_dyn = "l2_rolling"
tube_dyn = "NN_oneshot"
# nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/k1kfktrl"  # 128x128 ReLU
nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/932hlryb"  # 128x128 softplus b=5
# nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/0i2o675r"  # 128x128 softplus b=5 hopper

time_it = True
H = 50
max_iter = 200

def arr2list(d):
    if type(d) is dict:
        return {key: arr2list(val) for key, val in d.items()}
    elif type(d) is np.ndarray:
        return [arr2list(v) for v in list(d)]
    elif type(d) is np.float64:
        return float(d)
    else:
        return d


def main():
    model_name = f'{nn_path}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)

    run_id = model_cfg.dataset.wandb_experiment
    with open(f"../rom_tracking_data/{run_id}/config.pickle", 'rb') as f:
        dataset_cfg = pickle.load(f)
    dataset_cfg = unnormalize_dict(dataset_cfg)
    dataset_cfg['env_config']['env']['num_envs'] = 1
    dataset_cfg['env_config']['trajectory_generator'] = {
        'cls': 'ClosedLoopTrajectoryGenerator',
        'H': H,
        'dt_loop': dataset_cfg['env_config']['env']['model']['dt'],
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'prob_dict': {key: arr2list(val) for key, val in problem_dict[prob_str].items()},
        'tube_dyn': tube_dyn,
        'nn_path': nn_path,
        'w_max': 1,
        'warm_start': warm_start,
        'nominal_ws': 'interpolate',
        'track_nominal': track_warm,
        'tube_ws': tube_ws,
        'max_iter': max_iter
    }
    dataset_cfg['env_config']['domain_rand']['randomize_rom_distance'] = False

    dataset_cfg = OmegaConf.create(unnormalize_dict(dataset_cfg))

    # Define a new custom sim
    env = CustomSim(dataset_cfg.env_config)
    controller = instantiate(dataset_cfg.controller)(state_dependent_input_bound=env.model.clip_v_z)


    tube_ws_str = str(tube_ws).replace('.', '_')
    z_k = torch.zeros((H + 1, env.traj_gen.planning_model.n), device=env.device) * torch.nan
    v_k = torch.zeros((H, env.traj_gen.planning_model.m), device=env.device) * torch.nan
    x = torch.zeros((1, z_k.shape[0], env.model.n), device=env.device) * torch.nan
    w_k = torch.zeros((H + 1, 1), device=env.device) * torch.nan
    pz_x = torch.zeros_like(z_k, device=env.device) * torch.nan

    # mats for visualizing later
    z_vis = torch.zeros((H, *z_k.shape), device=env.device)
    v_vis = torch.zeros((H, *v_k.shape), device=env.device)
    pz_x_vis = torch.zeros((H, *pz_x.shape), device=env.device)
    w_vis = torch.zeros((H, *w_k.shape), device=env.device)
    z_sol_vis = torch.zeros((H, env.traj_gen.N + 1, env.traj_gen.planning_model.n), device=env.device)
    v_sol_vis = torch.zeros((H, env.traj_gen.N, env.traj_gen.planning_model.m), device=env.device)
    w_sol_vis = torch.zeros((H, env.traj_gen.N, 1), device=env.device)
    cv_vis = {}
    timing = np.zeros((H,))

    env.reset()
    z_k[0, :] = env.traj_gen.get_trajectory()[:, 0, :]
    x[:, 0, :] = env.get_states()
    pz_x[0, :] = env.model.proj_z(x[:, 0, :])
    w_k[0] = torch.linalg.norm(z_k[0, :] - pz_x[0, :])

    obs = env.get_observations()
    t0 = time.perf_counter_ns()

    for t in range(H):
        # Step environment until rom steps
        k = torch.clone(env.traj_gen.k.detach())
        while torch.any(env.traj_gen.k == k):
            actions = controller(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())

        # Save Data
        base = torch.clone(env.root_states.detach())
        d = torch.clone(dones.detach())
        proj = env.rom.proj_z(base)
        v_k[t, :] = env.traj_gen.v_trajectory[:, 0, :]
        x[:, t + 1, :] = env.get_states()
        z_k[t + 1, :] = env.traj_gen.get_trajectory()[:, 0, :]
        pz_x[t + 1, :] = proj
        w_k[t + 1] = env.traj_gen.w_trajectory[0].item()

        if not time_it:

            # Plot state space solution
            plt.subplot(3, 1, 1)
            plt.plot(env.traj_gen.trajectory[0].cpu().numpy())
            plt.legend(["x", "y"])
            plt.xlabel("Node")
            plt.ylabel("State")

            plt.subplot(3, 1, 2)
            plt.plot(env.traj_gen.v_trajectory[0].cpu().numpy())
            plt.legend(["v_x", "v_y"])
            plt.xlabel("Node")
            plt.ylabel("Input")

            plt.subplot(3, 1, 3)
            plt.plot(env.traj_gen.w_trajectory)
            plt.legend(["w"])
            plt.xlabel("Node")
            plt.ylabel("Tube")
            plt.show()

            # Plot constraint violation
            plt.figure()
            for label, data in env.traj_gen.g_dict.items():
                plt.plot(data, label=label)
            plt.title("Constraint Violation")
            plt.legend()
            plt.show()

            # Plot spacial solution
            fig, ax = plt.subplots()
            plot_problem(ax, env.traj_gen.obs, env.traj_gen.start, env.traj_gen.goal)
            env.traj_gen.planning_model.plot_tube(ax, z_k.cpu().numpy(), w_k.cpu().numpy(), 'k')
            env.traj_gen.planning_model.plot_spacial(ax, z_k.cpu().numpy(), '.-k')
            env.traj_gen.planning_model.plot_spacial(ax, pz_x.cpu().numpy(), '.-b')
            env.traj_gen.planning_model.plot_spacial(ax, env.traj_gen.trajectory[0].cpu().numpy())
            env.traj_gen.planning_model.plot_tube(ax, env.traj_gen.trajectory[0].cpu().numpy(), env.traj_gen.w_trajectory)
            plt.axis("square")
            plt.show()

        z_vis[t] = torch.clone(z_k)
        v_vis[t] = torch.clone(v_k)
        pz_x_vis[t] = torch.clone(pz_x)
        w_vis[t] = torch.clone(w_k)
        z_sol_vis[t] = torch.clone(env.traj_gen.trajectory)
        v_sol_vis[t] = torch.clone(env.traj_gen.v_trajectory)
        w_sol_vis[t] = torch.clone(torch.from_numpy(env.traj_gen.w_trajectory).float()).to(env.device)
        cv_vis["cv" + str(t)] = env.traj_gen.g_dict.copy()
        timing[t] = time.perf_counter_ns() - t0


    from scipy.io import savemat
    fn = f"data/cl_tube_{prob_str}_{nn_path[-8:]}_{warm_start}_{tube_dyn}_{tube_ws_str}_{track_warm}.mat"
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
        "obs_x": env.traj_gen.obs['c'][0, :],
        "obs_y": env.traj_gen.obs['c'][1, :],
        "obs_r": env.traj_gen.obs['r'],
        "timing": timing
    })

    print(f"Complete! Writing to {fn}")


if __name__ == '__main__':
    main()
