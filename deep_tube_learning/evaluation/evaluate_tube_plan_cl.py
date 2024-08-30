from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
from trajopt.tube_trajopt import *
import pickle
from omegaconf import OmegaConf
from deep_tube_learning.utils import unnormalize_dict
import time
import torch
from trajopt.rom_dynamics import SingleInt2D, DoubleInt2D

# prob_str = 'right'
# prob_str = 'right_wide'
# prob_str = 'gap'
prob_str = 'gap_big'

track_warm = False

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

time_it = False

def main(start, goal, obs, H):
    model_name = f'{nn_path}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)

    run_id = model_cfg.dataset.wandb_experiment
    with open(f"../rom_tracking_data/{run_id}/config.pickle", 'rb') as f:
        dataset_cfg = pickle.load(f)
    dataset_cfg = OmegaConf.create(unnormalize_dict(dataset_cfg))

    z_max = np.array([dataset_cfg.pos_max, dataset_cfg.pos_max])
    v_max = np.array([dataset_cfg.vel_max, dataset_cfg.vel_max])
    planning_model = CasadiSingleInt2D(dataset_cfg.env_config.rom.dt, -z_max, z_max, -v_max, v_max)
    pm = SingleInt2D(
        dataset_cfg.env_config.rom.dt, -z_max, z_max, -v_max, v_max, backend='torch'
    )
    dyn_cfg = dataset_cfg.env_config.env.model
    model_class = globals()[dyn_cfg.cls]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(
        dt=dyn_cfg.dt,
        z_min=torch.tensor(dyn_cfg.z_min, device=device),
        z_max=torch.tensor(dyn_cfg.z_max, device=device),
        v_min=torch.tensor(dyn_cfg.v_min, device=device),
        v_max=torch.tensor(dyn_cfg.v_max, device=device),
        n_robots=1,
        backend='torch',
        device=device
    )
    controller = instantiate(dataset_cfg.controller)(state_dependent_input_bound=model.clip_v_z)

    Q = 10 * np.eye(2)
    Qw = 0
    R = 10 * np.eye(2)
    N = model_cfg.dataset.H_fwd
    H_rev = model_cfg.dataset.H_rev
    w_max = 1

    tube_dynamics = get_tube_dynamics(tube_dyn, nn_path=nn_path)

    tube_ws_str = str(tube_ws).replace('.', '_')
    z_k = torch.zeros((H + 1, planning_model.n), device=device) * torch.nan
    v_k = torch.zeros((H, planning_model.m), device=device) * torch.nan
    e = np.zeros((H_rev, 1))
    v_prev = np.zeros((H_rev, planning_model.m))
    x = torch.zeros((1, z_k.shape[0], model.n), device=device) * torch.nan
    u = torch.zeros((1, v_k.shape[0], model.m), device=device) * torch.nan
    w_k = torch.zeros((H + 1, 1), device=device) * torch.nan
    pz_x = torch.zeros_like(z_k, device=device) * torch.nan
    z_k[0, :] = torch.from_numpy(start).float().to(device)
    x[:, 0, :2] = torch.from_numpy(start).float().to(device)
    x[:, 0, 2:] = 0
    pz_x[0, :] = model.proj_z(x[:, 0, :])

    # mats for visualizing later
    z_vis = torch.zeros((H, *z_k.shape), device=device)
    v_vis = torch.zeros((H, *v_k.shape), device=device)
    pz_x_vis = torch.zeros((H, *pz_x.shape), device=device)
    w_vis = torch.zeros((H, *w_k.shape), device=device)
    z_sol_vis = torch.zeros((H, N + 1, planning_model.n), device=device)
    v_sol_vis = torch.zeros((H, N, planning_model.m), device=device)
    w_sol_vis = torch.zeros((H, N, 1), device=device)
    cv_vis = {}
    timing = np.zeros((H,))
    t0 = time.perf_counter_ns()

    sol, solver = solve_tube(
        start, goal, obs, planning_model, tube_dynamics, N, H_rev, Q, Qw, R, w_max,
        warm_start=warm_start, tube_ws=tube_ws, max_iter=200, track_warm=track_warm
    )

    for k in range(H):
        z_sol, v_sol, w_sol = extract_solution(sol, N, planning_model.n, planning_model.m)

        # Decide fom action
        xt = x[:, k, :]
        for i in range(2):
            ut = controller(
                torch.concatenate((xt.squeeze(), torch.from_numpy(z_sol[0, :]).float().to(device), torch.from_numpy(v_sol[1, :]).float().to(device)))[None, :]
            )
            xt = model.f(xt, ut)
        v_k[k, :] = torch.from_numpy(v_sol[0, :]).float().to(device)
        z_k[k + 1, :] = pm.f(z_k[k, :][None, :], v_k[k, :][None, :])
        x[:, k + 1, :] = xt
        u[:, k, :] = ut
        pz_x[k + 1, :] = model.proj_z(xt)
        w_k[k + 1, :] = torch.from_numpy(w_sol[0, :]).float().to(device)

        g_violation = compute_constraint_violation(solver, sol["g"])
        g_dict = segment_constraint_violation(g_violation, solver["g_cols"])

        if not time_it:

            # Plot state space solution
            plt.subplot(3, 1, 1)
            plt.plot(z_sol)
            plt.legend(["x", "y"])
            plt.xlabel("Node")
            plt.ylabel("State")

            plt.subplot(3, 1, 2)
            plt.plot(v_sol)
            plt.legend(["v_x", "v_y"])
            plt.xlabel("Node")
            plt.ylabel("Input")

            plt.subplot(3, 1, 3)
            plt.plot(w_sol)
            plt.legend(["w"])
            plt.xlabel("Node")
            plt.ylabel("Tube")
            plt.show()

            # Plot constraint violation
            plt.figure()
            for label, data in g_dict.items():
                plt.plot(data, label=label)
            plt.title("Constraint Violation")
            plt.legend()
            plt.show()

            # Plot spacial solution
            fig, ax = plt.subplots()
            plot_problem(ax, obs, start, goal)
            planning_model.plot_tube(ax, z_k.cpu().numpy(), w_k.cpu().numpy(), 'k')
            planning_model.plot_spacial(ax, z_k.cpu().numpy(), '.-k')
            planning_model.plot_spacial(ax, pz_x.cpu().numpy(), '.-b')
            planning_model.plot_spacial(ax, z_sol)
            planning_model.plot_tube(ax, z_sol, w_sol)
            plt.axis("square")
            plt.show()

        z_vis[k] = torch.clone(z_k)
        v_vis[k] = torch.clone(v_k)
        pz_x_vis[k] = torch.clone(pz_x)
        w_vis[k] = torch.clone(w_k)
        z_sol_vis[k] = torch.clone(torch.from_numpy(z_sol).float())
        v_sol_vis[k] = torch.clone(torch.from_numpy(v_sol).float())
        w_sol_vis[k] = torch.clone(torch.from_numpy(w_sol).float())
        cv_vis["cv" + str(k)] = g_dict.copy()
        timing[k] = time.perf_counter_ns() - t0

        params = init_params(z_k[k + 1, :].detach().cpu().numpy(), goal, obs)
        e[:-1] = e[1, :]
        e[-1] = np.linalg.norm((z_k[k, :] - pz_x[k, :]).detach().cpu().numpy())
        v_prev[:-1, :] = v_prev[1:, :]
        v_prev[-1, :] = v_k[k, :].detach().cpu().numpy()
        params = np.vstack([params, e, v_prev.reshape(-1, 1)])
        x_init = init_decision_var(z_sol, v_sol, w=w_sol)

        sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"],
                               ubx=solver["ubx"])

    from scipy.io import savemat
    fn = f"data/cl_tube_{prob_str}_{warm_start}_{tube_dyn}_{tube_ws_str}.mat"
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
        "z0": start,
        "zf": goal,
        "obs_x": obs['c'][0, :],
        "obs_y": obs['c'][1, :],
        "obs_r": obs['r'],
        "timing": timing
    })

    print(f"Complete! Writing to {fn}")


if __name__ == '__main__':
    main(
        problem_dict[prob_str]["start"],
        problem_dict[prob_str]["goal"],
        problem_dict[prob_str]["obs"],
        problem_dict[prob_str]["H"]
    )
