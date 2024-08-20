import time
import csv
import numpy as np
import pandas as pd
import casadi as ca
import l4casadi as l4c
import matplotlib.pyplot as plt
import wandb
from hydra.utils import instantiate
from deep_tube_learning.utils import wandb_model_load, wandb_model_load_cpu
from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
from trajopt.rom_planning import trajopt_solver as trajopt_solver_no_tube


start = np.array([0, 0])
goal = np.array([4, 3])

vel_max = 1    # m/x
pos_max = 10   # m
dt = 0.1

# obs = {
#     'c': np.array([[2, 3.5, 1, 3], [1.5, 0.5, 3, 3]]),
#     'r': np.array([0.1, 0.05, 0.05, 0.03])
# }

obs = {
    'c': np.array([[2, 0.], [1.5, 3]]),
    'r': np.array([1., 1])
}


def add_obs(k, z, w, p_obs_c, p_obs_r, Nobs, g, g_lb, g_ub):
    for i in range(Nobs):
        # Simply increase distance from center by tube dimension
        g = ca.horzcat(g, ca.sum2((z[k, :2] - p_obs_c[i, :]) ** 2) - (p_obs_r[i, :] + w[k, :]) ** 2)
        g_lb = ca.horzcat(g_lb, ca.DM([0]))
        g_ub = ca.horzcat(g_ub, ca.DM.inf())
    return g, g_lb, g_ub


# TODO: tube solver
def trajopt_tube_solver(pm, tube_oneshot_model, w_max, N, Q, Qw, R, Nobs, Qf=None, device='cpu'):
    if Qf is None:
        Qf = Q
    Q = ca.DM(Q)
    Qf = ca.DM(Qf)
    z_min = ca.DM(pm.z_min)
    z_max = ca.DM(pm.z_max)
    v_min = ca.DM(pm.v_min)
    v_max = ca.DM(pm.v_max)

    fw = l4c.L4CasADi(tube_oneshot_model, device=device)

    # Make decision variables (2D double integrator)
    z = ca.MX.sym("z", N + 1, pm.n)
    v = ca.MX.sym("v", N, pm.m)
    w = ca.MX.sym("w", N + 1, 1)

    # Parameters: initial condition, final condition
    p_z0 = ca.MX.sym("p_z0", 1, pm.n)          # Initial projection Pz(x0) state
    p_zf = ca.MX.sym("p_zf", 1, pm.n)          # Goal state
    p_obs_c = ca.MX.sym("p_obs_c", Nobs, 2)    # positional obstacle centers
    p_obs_r = ca.MX.sym("p_obs_r", Nobs, 1)    # positional obstacle radii

    # Define NLP
    obj = 0
    g = []
    g_lb = []
    g_ub = []

    z_lb = ca.repmat(z_min.T, N + 1, 1)
    z_ub = ca.repmat(z_max.T, N + 1, 1)
    v_lb = ca.repmat(v_min.T, N, 1)
    v_ub = ca.repmat(v_max.T, N, 1)
    w_lb = ca.DM(N + 1, 1)
    w_ub = ca.DM(np.ones((N + 1, 1)) * w_max)

    for k in range(N):
        # cost function
        obj += (z[k, :] - p_zf) @ Q @ (z[k, :] - p_zf).T + v[k, :] @ R @ v[k, :].T + Qw * w[k, :]**2

        # dynamics
        g = ca.horzcat(g, pm.f(z[k, :].T, v[k, :].T).T - z[k + 1, :])
        g_lb = ca.horzcat(g_lb, ca.DM(np.zeros((pm.n,))).T)
        g_ub = ca.horzcat(g_ub, ca.DM(np.zeros((pm.n,))).T)

        # obstacle constraints
        g, g_lb, g_ub = add_obs(k, z, w, p_obs_c, p_obs_r, Nobs, g, g_lb, g_ub)

    # Tube dynamics
    tube_input = ca.horzcat(w[0, :], z[0, 2:], ca.reshape(v, 1, v.numel()))

    def fw_tmp(input):
        v = ca.reshape(input[1:], -1, 2)
        # w = 0.5 * (v[:, 0]**2 + v[:, 1]**2)
        w = 0.5 * (ca.fabs(v[:, 0]) + ca.fabs(v[:, 1]))
        # w = 0.1 * ca.DM(np.ones(w.shape))
        return w

    def fw_tmp_rolling(input):
        v = ca.reshape(input[1:], -1, 2)
        # w = 0.5 * (v[:, 0] ** 2 + v[:, 1] ** 2)
        w = 0.5 * (ca.fabs(v[:, 0]) + ca.fabs(v[:, 1]))
        window_size = 10

        # Initialize a list to store the rolling averages
        rolling_avg = [ca.sum1(w[max(i - window_size + 1, 0):i + 1]) / min(window_size, i + 1) for i in
                       range(w.numel())]

        return ca.vertcat(*rolling_avg)

    g = ca.horzcat(g, fw_tmp(tube_input.T).T - w[1:, :].T)
    # g = ca.horzcat(g, fw_tmp_rolling(tube_input.T).T - w[1:, :].T)
    # g = ca.horzcat(g, fw(tube_input.T).T - w[1:, :].T)
    g_lb = ca.horzcat(g_lb, ca.DM(np.zeros((H,))).T)
    g_ub = ca.horzcat(g_ub, ca.DM(np.zeros((H,))).T)
    # delta = .75
    # g_lb = ca.horzcat(g_lb, ca.DM(-delta * np.ones((H,))).T)
    # g_ub = ca.horzcat(g_ub, ca.DM(delta * np.ones((H,))).T)

    # Terminal cost/constraints
    obj += (z[N, :] - p_zf) @ Qf @ (z[N, :] - p_zf).T + Qw * w[N, :]**2
    g, g_lb, g_ub = add_obs(N, z, w, p_obs_c, p_obs_r, Nobs, g, g_lb, g_ub)

    # Initial condition
    g = ca.horzcat(g, w[0, :]**2 - ca.dot(z[0, :] - p_z0, z[0, :] - p_z0))
    g_lb = ca.horzcat(g_lb, ca.DM(np.zeros((1,))).T)
    g_ub = ca.horzcat(g_ub, ca.DM(np.ones((1,)) * (w_max**2)).T)

    # Generate solver
    x_nlp = ca.vertcat(
        ca.reshape(z, (N + 1) * pm.n, 1),
        ca.reshape(v, N * pm.m, 1),
        ca.reshape(w, N + 1, 1)
    )
    lbx = ca.vertcat(
        ca.reshape(z_lb, (N + 1) * pm.n, 1),
        ca.reshape(v_lb, N * pm.m, 1),
        ca.reshape(w_lb, N + 1, 1)
    )
    ubx = ca.vertcat(
        ca.reshape(z_ub, (N + 1) * pm.n, 1),
        ca.reshape(v_ub, N * pm.m, 1),
        ca.reshape(w_ub, N + 1, 1)
    )
    p_nlp = ca.vertcat(p_z0.T, p_zf.T, ca.reshape(p_obs_c, 2 * Nobs, 1), p_obs_r)
    nlp_dict = {
        "x": x_nlp,
        "f": obj,
        "g": g,
        "p": p_nlp
    }
    nlp_opts = {
        "ipopt.linear_solver": "mumps",
        "ipopt.sb": "yes",
        "ipopt.max_iter": 10000,
        "ipopt.tol": 1e-4,
        # "ipopt.print_level": 5,
        "print_time": True,
    }

    nlp_solver = ca.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx}

    return solver, nlp_dict, nlp_opts


def generate_trajectory(plan_model, z0, zf, tube_oneshot_model, w_max, N, Q, Qw, R, Qf=None, device='cpu'):
    Nobs = len(obs['r'])

    solver, nlp_dict, nlp_opts = trajopt_tube_solver(plan_model, tube_oneshot_model, w_max, N, Q, Qw, R, Nobs, Qf=Qf, device=device)
    nlp_no_tube = trajopt_solver_no_tube(plan_model, N, Q, R, Nobs)

    params = np.vstack([z0[:, None], zf[:, None], np.reshape(obs['c'], (2 * Nobs, 1)), obs['r'][:, None]])

    v_init = np.zeros((N, plan_model.m))
    # z_init = np.repeat(zf[:, None], N + 1, 1)
    # z_init = np.repeat(z0[:, None], N + 1, 1)
    z_init = np.outer(np.linspace(0, 1, N + 1), (zf - z0)) + z0
    w_init = np.zeros((N + 1, 1))

    x_init = np.vstack([
        np.reshape(z_init, ((N + 1) * plan_model.n, 1), order='F'),
        np.reshape(v_init, (N * plan_model.m, 1), order='F')
    ])

    sol_no_tube = nlp_no_tube["solver"](x0=x_init, p=params, lbg=nlp_no_tube["lbg"], ubg=nlp_no_tube["ubg"],
                                        lbx=nlp_no_tube["lbx"], ubx=nlp_no_tube["ubx"])
    z_ind = (N + 1) * plan_model.n
    v_ind = N * plan_model.m
    z_init = np.array(sol_no_tube["x"][:z_ind, :].reshape((N + 1, plan_model.n)))
    v_init = np.array(sol_no_tube["x"][z_ind:z_ind + v_ind, :].reshape((N, plan_model.m)))
    x_init = np.vstack([
        np.reshape(z_init, ((N + 1) * plan_model.n, 1), order='F'),
        np.reshape(v_init, (N * plan_model.m, 1), order='F'),
        np.reshape(w_init, (N + 1, 1), order='F')
    ])


    # now the Tube
    nlp_opts["ipopt.max_iter"] = 1000
    nlp_solver = ca.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

    solver["solver"] = nlp_solver
    tic = time.perf_counter_ns()
    sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"], ubx=solver["ubx"])
    toc = time.perf_counter_ns()
    print(f"Solve Time: {(toc - tic) / 1e6}ms")

    # extract solution
    z_ind = (N + 1) * plan_model.n
    v_ind = N * plan_model.m
    z_sol = np.array(sol["x"][:z_ind, :].reshape((N + 1, plan_model.n)))
    v_sol = np.array(sol["x"][z_ind:z_ind + v_ind, :].reshape((N, plan_model.m)))
    w_sol = np.array(sol["x"][z_ind + v_ind:, :].reshape((N + 1, 1)))

    fig, axs = plt.subplots(2, 1)
    plan_model.plot_ts(axs, z_sol, v_sol)
    plt.show()

    fig, ax = plt.subplots()
    for i in range(Nobs):
        xc = obs['c'][0, i]
        yc = obs['c'][1, i]
        circ = plt.Circle((xc, yc), obs['r'][i], color='r', alpha=0.5)
        ax.add_patch(circ)
    plt.plot(z0[0], z0[1], 'rx')
    plt.plot(zf[0], zf[1], 'go')

    plan_model.plot_tube(ax, z_sol, w_sol)
    plan_model.plot_spacial(ax, z_init, 'k')
    plan_model.plot_spacial(ax, z_sol)
    plt.axis("square")
    plt.show()

    print("Now saving iterations")


    cols = ["iter"]
    for t in range(N + 1):
        cols.extend([f"z_x{t}", f"z_y{t}"])
    for t in range(N):
        cols.extend([f"v_x{t}", f"v_y{t}"])
    for t in range(N + 1):
        cols.append(f"w{t}")
    for t in range(solver["lbg"].numel()):
        cols.append(f"g_{t}")
    for t in range(solver["lbg"].numel()):
        cols.append(f"lbg{t}")
    for t in range(solver["lbg"].numel()):
        cols.append(f"ubg{t}")
    for t in range(solver["lbx"].numel()):
        cols.append(f"lbx{t}")
    for t in range(solver["lbx"].numel()):
        cols.append(f"ubx{t}")

    df = pd.DataFrame(columns=cols)
    tic = time.perf_counter_ns()
    for it in range(0, 1000, 10):
        nlp_opts["ipopt.max_iter"] = it
        nlp_solver = ca.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

        solver["solver"] = nlp_solver
        sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"], ubx=solver["ubx"])
        toc = time.perf_counter_ns()
        print(f"Solve Time: {(toc - tic) / 1e6}ms")

        # extract solution
        z_ind = (N + 1) * plan_model.n
        v_ind = N * plan_model.m
        z_sol = np.array(sol["x"][:z_ind, :].reshape((N + 1, plan_model.n)))
        v_sol = np.array(sol["x"][z_ind:z_ind + v_ind, :].reshape((N, plan_model.m)))
        w_sol = np.array(sol["x"][z_ind + v_ind:, :].reshape((N + 1, 1)))

        # TODO: Debug warm start
        g = np.array(solver["solver"].get_function("nlp_g")(sol['x'], params)).T
        ubg = np.array(solver["ubg"]).T
        lbg = np.array(solver["lbg"]).T

        lbx = np.array(solver["lbx"]).T
        ubx = np.array(solver["ubx"]).T

        new_row = np.concatenate((
            np.array([it]),
            z_sol.reshape((-1,)),
            v_sol.reshape((-1,)),
            w_sol.squeeze(),
            g.squeeze(),
            lbg.squeeze(),
            ubg.squeeze(),
            lbx.squeeze(),
            ubx.squeeze()
        ), axis=-1)
        new_row_df = pd.DataFrame([new_row], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)

    df.to_csv("debug_trajopt_results.csv", index=False)
    print("Complete")


if __name__ == "__main__":
    z_max = np.array([pos_max, pos_max])
    v_max = np.array([vel_max, vel_max])
    planning_model = CasadiSingleInt2D(dt, -z_max, z_max, -v_max, v_max)

    Q = 10 * np.eye(2)
    Qw = 0.5
    R = 0.1 * np.eye(2)
    z0 = start
    zf = goal
    device = 'cuda'
    w_max = 1

    exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/3vdx800j"  # 256x256, H=50
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/yfofrrk1"  #32x32, H=5
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/trq7kcv2"  # 128x128 softplus
    model_name = f'{exp_name}_model:best'

    api = wandb.Api()
    # model_cfg, state_dict = wandb_model_load(api, model_name)
    model_cfg, state_dict = wandb_model_load_cpu(api, model_name)
    H = model_cfg.dataset.H
    tube_oneshot_model = instantiate(model_cfg.model)(1 + 2 * H, H)

    tube_oneshot_model.load_state_dict(state_dict)

    tube_oneshot_model.to(device)
    tube_oneshot_model.eval()

    generate_trajectory(planning_model, z0, zf, tube_oneshot_model, w_max, H, Q, Qw, R, device=device)
