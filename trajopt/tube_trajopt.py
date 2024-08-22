import wandb
import numpy as np
import pandas as pd
import casadi as ca
import l4casadi as l4c
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from deep_tube_learning.utils import wandb_model_load, wandb_model_load_cpu


problem_dict = {
    "gap": {"start": np.array([0, 0]), "goal": np.array([3., 3]),
            "obs": {'c': np.array([[2, 0.], [1.5, 3]]), 'r': np.array([1., 1])},
            "vel_max": 1, "pos_max": 10, "dt": 0.1},
    "right": {"start": np.array([0, 0]), "goal": np.array([4, 0]),
              "obs": {'c': np.array([[2, 2.], [1.25, -1.25]]), 'r': np.array([1., 1])},
              "vel_max": 1, "pos_max": 10, "dt": 0.1},
    "right_wide": {"start": np.array([0, 0]), "goal": np.array([4, 0]),
                   "obs": {'c': np.array([[2, 2.], [2.5, -2.5]]), 'r': np.array([1., 1])},
                   "vel_max": 1, "pos_max": 10, "dt": 0.1}
}


def dynamics_constraint(f, z, v):
    """
    Creates dynamics constraints
    :param f: dynamics function
    :param z: state variable, N+1 x n
    :param v: input variable, N x m
    :return: dynamics constraints, 1 x N*n
    """
    g = []
    for k in range(v.shape[0]):
        g.append(f(z[k, :].T, v[k, :].T).T - z[k + 1, :])
    g = ca.horzcat(*g)
    g_lb = ca.DM(*g.shape)
    g_ub = ca.DM(*g.shape)
    return g, g_lb, g_ub


def quadratic_objective(x, Q, goal=None):
    """
    Returns a quadratic objective function of the form sum_0^N x_k^T @ Q @ x_k
    :param x: state variable, N+1 x n
    :param Q: Quadratic cost matrix
    :param goal: desired state, 1 x n
    :return: scalar quadratic objective function
    """
    if goal is None:
        dist = x
    else:
        dist = x - ca.repmat(goal, x.shape[0], 1)
    return ca.sum1(ca.sum2((dist @ Q) * dist))


def single_obstacle_constraint_k(z, obs_c, obs_r):
    dist = z[:2] - obs_c
    g = ca.dot(dist, dist) - obs_r**2
    return g, ca.DM([0]), ca.DM.inf()


def single_obstacle_constraint(z, obs_c, obs_r, w=None):
    g = []
    g_lb = []
    g_ub = []
    for k in range(z.shape[0]):
        if w is None:
            g_k, g_lb_k, g_ub_k = single_obstacle_constraint_k(z[k, :], obs_c, obs_r)
        else:
            g_k, g_lb_k, g_ub_k = single_obstacle_constraint_k(z[k, :], obs_c, obs_r + w[k])
        g.append(g_k)
        g_lb.append(g_lb_k)
        g_ub.append(g_ub_k)
    return ca.horzcat(*g), ca.horzcat(*g_lb), ca.horzcat(*g_ub)


def obstacle_constraints(z, obs_c, obs_r, w=None):
    """
    Constructs circular obstacle constraints, for obstacles of given centers and radii. If w (tube diameter)
    is given, the radius of the tube is expanded by this amount.
    :param z: state variable, N+1 x n
    :param obs_c: center of obstacle, K x 2
    :param obs_r: radius of obstacle, K x 1
    :param w: tube diameter, added to radius if given. Default None.
    """
    g = []
    g_lb = []
    g_ub = []
    for i in range(obs_r.shape[0]):
        g_i, g_lb_i, g_ub_i = single_obstacle_constraint(z, obs_c[i, :], obs_r[i, :], w=w)
        g.append(g_i)
        g_lb.append(g_lb_i)
        g_ub.append(g_ub_i)
    return ca.horzcat(*g), ca.horzcat(*g_lb), ca.horzcat(*g_ub)


def initial_condition_equality_constraint(z, z0):
    dist = z[0, :2] - z0
    return dist, ca.DM(*dist.shape), ca.DM(*dist.shape)


def setup_trajopt_solver(pm, N, Nobs):
    # Set up state/input bounds
    z_min = ca.DM(pm.z_min)
    z_max = ca.DM(pm.z_max)
    v_min = ca.DM(pm.v_min)
    v_max = ca.DM(pm.v_max)
    z_lb = ca.repmat(z_min.T, N + 1, 1)
    z_ub = ca.repmat(z_max.T, N + 1, 1)
    v_lb = ca.repmat(v_min.T, N, 1)
    v_ub = ca.repmat(v_max.T, N, 1)

    # Parameters: initial condition, final condition
    p_z0 = ca.MX.sym("p_z0", 1, pm.n)  # Initial projection Pz(x0) state
    p_zf = ca.MX.sym("p_zf", 1, pm.n)  # Goal state
    p_obs_c = ca.MX.sym("p_obs_c", Nobs, 2)  # positional obstacle centers
    p_obs_r = ca.MX.sym("p_obs_r", Nobs, 1)  # positional obstacle radii

    # Make decision variables (2D double integrator)
    z = ca.MX.sym("z", N + 1, pm.n)
    v = ca.MX.sym("v", N, pm.m)

    return z, v, z_lb, z_ub, v_lb, v_ub, p_z0, p_zf, p_obs_c, p_obs_r


def trajopt_solver(pm, N, Q, R, Nobs, Qf=None, max_iter=1000, debug_filename=None):
    z, v, z_lb, z_ub, v_lb, v_ub, p_z0, p_zf, p_obs_c, p_obs_r = setup_trajopt_solver(pm, N, Nobs)

    if Qf is None:
        Qf = Q
    Q = ca.DM(Q)
    Qf = ca.DM(Qf)

    # Define NLP
    obj = quadratic_objective(z[:-1, :], Q, goal=p_zf) + quadratic_objective(v, R) + quadratic_objective(z[-1, :], Qf, goal=p_zf)
    g_dyn, g_lb_dyn, g_ub_dyn = dynamics_constraint(pm.f, z, v)
    g_obs, g_lb_obs, g_ub_obs = obstacle_constraints(z, p_obs_c, p_obs_r)
    g_ic, g_lb_ic, g_ub_ic = initial_condition_equality_constraint(z, p_z0)

    g = ca.horzcat(g_dyn, g_obs, g_ic)
    g_lb = ca.horzcat(g_lb_dyn, g_lb_obs, g_lb_ic)
    g_ub = ca.horzcat(g_ub_dyn, g_ub_obs, g_ub_ic)

    g_dyn_cols = []
    for k in range(N):
        g_dyn_cols.extend(["dyn_" + st + f"_{k}" for st in pm.state_names])
    g_obs_cols = []
    for i in range(Nobs):
        g_obs_cols.extend([f"obs_{i}_{k}" for k in range(N + 1)])
    g_cols = g_dyn_cols + g_obs_cols + ["ic_" + st for st in pm.state_names]
    z_str = np.array(['z'] * ((N + 1) * pm.n)).reshape(z.shape)
    v_str = np.array(['v'] * (N * pm.m)).reshape(v.shape)

    # Generate solver
    x_nlp = ca.vertcat(
        ca.reshape(z, (N + 1) * pm.n, 1),
        ca.reshape(v, N * pm.m, 1),
    )
    x_cols = list(np.vstack((
        np.reshape(z_str, ((N + 1) * pm.n, 1)),
        np.reshape(v_str, (N * pm.m, 1))
    )))
    lbx = ca.vertcat(
        ca.reshape(z_lb, (N + 1) * pm.n, 1),
        ca.reshape(v_lb, N * pm.m, 1),
    )
    ubx = ca.vertcat(
        ca.reshape(z_ub, (N + 1) * pm.n, 1),
        ca.reshape(v_ub, N * pm.m, 1),
    )
    p_nlp = ca.vertcat(p_z0.T, p_zf.T, ca.reshape(p_obs_c, 2 * Nobs, 1), p_obs_r)
    obs_c_lst = []
    obs_c_lst = [f'obs_{i}_x' for i in range(Nobs)] + [f'obs_{i}_y' for i in range(Nobs)]
    p_cols = [f'z_ic_{i}' for i in range(pm.n)] + [f'z_g_{i}' for i in range(pm.n)] + obs_c_lst + [f'obs_{i}_r' for i in
                                                                                                   range(Nobs)]
    nlp_dict = {
        "x": x_nlp,
        "f": obj,
        "g": g,
        "p": p_nlp
    }
    nlp_opts = {
        "ipopt.linear_solver": "mumps",
        "ipopt.sb": "yes",
        "ipopt.max_iter": max_iter,
        "ipopt.tol": 1e-4,
        # "ipopt.print_level": 5,
        "print_time": True,
    }

    if debug_filename is not None:
        nlp_opts['iteration_callback'] = SolverCallback('iter_callback', debug_filename, x_cols, g_cols, p_cols, {})

    nlp_solver = ca.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx, "g_cols": g_cols, "x_cols": x_cols, "p_cols": p_cols}

    return solver, nlp_dict, nlp_opts


def trajopt_tube_solver(pm, tube_dynamics, N, Q, Qw, R, w_max, Nobs, Qf=None, max_iter=1000, debug_filename=None):
    z, v, z_lb, z_ub, v_lb, v_ub, p_z0, p_zf, p_obs_c, p_obs_r = setup_trajopt_solver(pm, N, Nobs)
    w = ca.MX.sym("w", N + 1, 1)
    w_lb = ca.DM(N + 1, 1)
    w_ub = ca.DM(np.ones((N + 1, 1)) * w_max)

    if Qf is None:
        Qf = Q
    Q = ca.DM(Q)
    Qf = ca.DM(Qf)

    # Define NLP
    obj = quadratic_objective(z[:-1, :], Q, goal=p_zf) + quadratic_objective(v, R) + \
          quadratic_objective(z[-1, :], Qf, goal=p_zf) + quadratic_objective(w, Qw)
    g_dyn, g_lb_dyn, g_ub_dyn = dynamics_constraint(pm.f, z, v)
    g_obs, g_lb_obs, g_ub_obs = obstacle_constraints(z, p_obs_c, p_obs_r, w=w)
    g_ic, g_lb_ic, g_ub_ic = initial_condition_equality_constraint(z, p_z0)
    g_tube, g_lb_tube, g_ub_tube = tube_dynamics(z, v, w)

    g = ca.horzcat(g_dyn, g_obs, g_ic, g_tube)
    g_lb = ca.horzcat(g_lb_dyn, g_lb_obs, g_lb_ic, g_lb_tube)
    g_ub = ca.horzcat(g_ub_dyn, g_ub_obs, g_ub_ic, g_ub_tube)
    g = g.T
    g_lb = g_lb.T
    g_ub = g_ub.T

    # Generate solver
    x_nlp = ca.vertcat(
        ca.reshape(z, (N + 1) * pm.n, 1),
        ca.reshape(v, N * pm.m, 1),
        w
    )

    lbx = ca.vertcat(
        ca.reshape(z_lb, (N + 1) * pm.n, 1),
        ca.reshape(v_lb, N * pm.m, 1),
        w_lb
    )
    ubx = ca.vertcat(
        ca.reshape(z_ub, (N + 1) * pm.n, 1),
        ca.reshape(v_ub, N * pm.m, 1),
        w_ub
    )
    p_nlp = ca.vertcat(p_z0.T, p_zf.T, ca.reshape(p_obs_c, 2 * Nobs, 1), p_obs_r)

    x_cols, g_cols, p_cols = generate_col_names(pm, N, Nobs, x_nlp, g, p_nlp)
    nlp_dict = {
        "x": x_nlp,
        "f": obj,
        "g": g,
        "p": p_nlp
    }
    nlp_opts = {
        "ipopt.linear_solver": "mumps",
        # "ipopt.linear_solver": "ma27",  # Library loading failure
        "ipopt.sb": "yes",
        "ipopt.max_iter": max_iter,
        "ipopt.tol": 1e-4,
        "print_time": True,
        # "ipopt.print_level": 5,
        # "ipopt.output_file": "data/ipopt_output.txt",
        # "ipopt.file_print_level": 8,
        "ipopt.jacobian_approximation": "exact",  # "exact" (default) or "finite-difference-values"
        "ipopt.gradient_approximation": "exact",  # "exact" (default) or "finite-difference-values"
        # "ipopt.max_wall_time": 0.1,
        # "ipopt.constr_viol_tol": 1e-1,
        "ipopt.hessian_approximation": "limited-memory"  # Seems to help get constraint violation down, now mostly dual infeasibility remains
    }

    if debug_filename is not None:
        nlp_opts['iteration_callback'] = SolverCallback('iter_callback', debug_filename, x_cols, g_cols, p_cols, {})

    nlp_solver = ca.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

    solver = {
        "solver": nlp_solver, "callback": nlp_opts['iteration_callback'],
        "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx,
        "g_cols": g_cols, "x_cols": x_cols, "p_cols": p_cols
    }

    return solver, nlp_dict, nlp_opts


def generate_col_names(pm, N, Nobs, x, g, p):
    z_str = np.array(["z"] * ((N + 1) * pm.n), dtype='U8').reshape((N + 1, pm.n))
    v_str = np.array(["v"] * (N * pm.m), dtype='U8').reshape((N, pm.m))
    for r in range(z_str.shape[0]):
        for c in range(z_str.shape[1]):
            z_str[r, c] = z_str[r, c] + f"_{r}_{c}"
    for r in range(v_str.shape[0]):
        for c in range(v_str.shape[1]):
            v_str[r, c] = v_str[r, c] + f"_{r}_{c}"
    if x.numel() == z_str.size + v_str.size:
        x_cols = list(np.vstack((
            np.reshape(z_str, ((N + 1) * pm.n, 1)),
            np.reshape(v_str, (N * pm.m, 1))
        )).squeeze())
    else:
        w_str = np.array(["w"] * (N + 1), dtype='U8').reshape((N + 1, 1))
        for r in range(w_str.shape[0]):
            w_str[r, 0] = w_str[r, 0] + f"_{r}"
        x_cols = list(np.vstack((
            np.reshape(z_str, ((N + 1) * pm.n, 1)),
            np.reshape(v_str, (N * pm.m, 1)),
            w_str
        )).squeeze())

    g_dyn_cols = []
    for k in range(N):
        g_dyn_cols.extend(["dyn_" + st + f"_{k}" for st in pm.state_names])
    g_obs_cols = []
    for i in range(Nobs):
        g_obs_cols.extend([f"obs_{i}_{k}" for k in range(N + 1)])
    g_tube_dyn = [f"tube_{k}" for k in range(N)]
    g_cols = g_dyn_cols + g_obs_cols + ["ic_" + st for st in pm.state_names]

    if not len(g_cols) == g.numel():
        g_cols.extend(g_tube_dyn)

    obs_c_lst = [f'obs_{i}_x' for i in range(Nobs)] + [f'obs_{i}_y' for i in range(Nobs)]
    p_cols = [f'z_ic_{i}' for i in range(pm.n)] + [f'z_g_{i}' for i in range(pm.n)] + obs_c_lst + \
             [f'obs_{i}_r' for i in range(Nobs)]

    assert len(x_cols) == x.numel() and len(g_cols) == g.numel() and len(p_cols) == p.numel()
    return x_cols, g_cols, p_cols


def init_params(z0, zf, obs):
    Nobs = len(obs['r'])
    params = np.vstack([z0[:, None], zf[:, None], np.reshape(obs['c'], (2 * Nobs, 1)), obs['r'][:, None]])
    return params


def init_decision_var(z, v, w=None):
    N = v.shape[0]
    n = z.shape[1]
    m = v.shape[1]
    if w is None:
        x_init = np.vstack([
            np.reshape(z, ((N + 1) * n, 1), order='F'),
            np.reshape(v, (N * m, 1), order='F')
        ])
    else:
        x_init = np.vstack([
            np.reshape(z, ((N + 1) * n, 1), order='F'),
            np.reshape(v, (N * m, 1), order='F'),
            np.reshape(w, ((N + 1) * 1, 1), order='F')
        ])
    return x_init


def extract_solution(sol, N, n, m):
    z_ind = (N + 1) * n
    v_ind = N * m
    z_sol = np.array(sol["x"][:z_ind, :].reshape((N + 1, n)))
    v_sol = np.array(sol["x"][z_ind:z_ind + v_ind, :].reshape((N, m)))
    if sol["x"].numel() > z_ind + v_ind:
        w_sol = np.array(sol["x"][z_ind + v_ind:, :].reshape((N + 1, 1)))
        return z_sol, v_sol, w_sol
    else:
        return z_sol, v_sol


def plot_problem(ax, obs, z0, zf):
    for i in range(len(obs["r"])):
        xc = obs['c'][0, i]
        yc = obs['c'][1, i]
        circ = plt.Circle((xc, yc), obs['r'][i], color='r', alpha=0.5)
        ax.add_patch(circ)
    plt.plot(z0[0], z0[1], 'rx')
    plt.plot(zf[0], zf[1], 'go')


def compute_constraint_violation(solver, g):
    g = np.array(g)
    ubg = np.array(solver["ubg"])
    lbg = np.array(solver["lbg"])
    viol = np.maximum(np.maximum(g - ubg, 0), np.maximum(lbg - g, 0))
    return viol


def segment_constraint_violation(g_viol, g_col):
    g_dyn_viol = g_viol[[j for j, s in enumerate(g_col) if "dyn" in s]]
    g_seg = {"Dynamics": g_dyn_viol}
    i = 0
    while i >= 0:
        idx = [j for j, s in enumerate(g_col) if f"obs_{i}" in s]
        if idx:
            g_obs_viol = g_viol[idx]
            g_seg[f"Obstacle {i}"] = g_obs_viol
            i += 1
        else:
            i = -1
    ic_viol = g_viol[[j for j, s in enumerate(g_col) if "ic" in s]]
    g_seg["Initial Condition"] = ic_viol

    tube_idx = [j for j, s in enumerate(g_col) if "tube" in s]
    if tube_idx:
        g_seg["Tube Dynamics"] = g_viol[tube_idx]

    return g_seg


def get_warm_start(warm_start, start, goal, N, planning_model, obs=None, Q=None, R=None, nominal_ws='interpolate'):
    if warm_start == 'start':
        v_init = np.zeros((N, planning_model.m))
        z_init = np.repeat(start[None, :], N + 1, 0)
    elif warm_start == 'goal':
        v_init = np.zeros((N, planning_model.m))
        z_init = np.repeat(goal[None, :], N + 1, 0)
    elif warm_start == 'interpolate':
        z_init = np.outer(np.linspace(0, 1, N + 1), (goal - start)) + start
        v_init = np.diff(z_init, axis=0) / planning_model.dt
    elif warm_start == 'nominal':
        assert obs is not None and Q is not None and R is not None
        sol, solver = solve_nominal(start, goal, obs, planning_model, N, Q, R, warm_start=nominal_ws)
        z_init, v_init = extract_solution(sol, N, planning_model.n, planning_model.m)
    else:
        raise ValueError(f'Warm start {warm_start} not implemented. Must be ic, goal, interpolate, or nominal')

    return z_init, v_init


def get_tube_warm_start(w_init, tube_dynamics, z, v, w):
    if w_init == "evaluate":
        return np.concatenate([np.array([[0]]), np.array(tube_dynamics(ca.DM(z), ca.DM(v), ca.DM(w))[0])], axis=-1).T
    elif isinstance(w_init, (int, float)):
        return np.ones((z.shape[0], 1)) * w_init
    raise ValueError(f"Tube warm start {w_init} not implemented. Must be evaluate or a double")


def solve_nominal(start, goal, obs, planning_model, N, Q, R, warm_start='start', debug_filename=None):
    solver, nlp_dict, nlp_opts = trajopt_solver(planning_model, N, Q, R, len(obs["r"]), debug_filename=debug_filename)

    z_init, v_init = get_warm_start(warm_start, start, goal, N, planning_model)

    params = init_params(start, goal, obs)
    x_init = init_decision_var(z_init, v_init)

    sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"],
                           ubx=solver["ubx"])

    if 'iteration_callback' in nlp_opts.keys():
        nlp_opts['iteration_callback'].write_data(solver, params)
    return sol, solver


def solve_tube(
        start, goal, obs, planning_model, tube_dynamics, N, Q, Qw, R, w_max, Qf=None,
        warm_start='start', nominal_ws='interpolate', tube_ws=0, debug_filename=None, max_iter=1000
):
    solver, nlp_dict, nlp_opts = trajopt_tube_solver(planning_model, tube_dynamics, N, Q, Qw, R, w_max, len(obs['r']), Qf=Qf, max_iter=max_iter, debug_filename=debug_filename)

    z_init, v_init = get_warm_start(warm_start, start, goal, N, planning_model, obs, Q, R, nominal_ws=nominal_ws)
    w_init = get_tube_warm_start(tube_ws, tube_dynamics, z_init, v_init, np.zeros((N + 1, 1)))

    params = init_params(start, goal, obs)
    x_init = init_decision_var(z_init, v_init, w=w_init)

    sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"],
                           ubx=solver["ubx"])

    if 'iteration_callback' in nlp_opts.keys():
        nlp_opts['iteration_callback'].write_data(solver, params)
    return sol, solver


def get_l1_tube_dynamics(scaling):

    def l1_tube_dyn(z, v, w):
        fw = scaling * ca.sum2(ca.fabs(v))
        g = (fw - w[1:]).T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    return l1_tube_dyn


def get_l2_tube_dynamics(scaling):

    def l2_tube_dyn(z, v, w):
        fw = scaling * ca.sum2(v ** 2)
        g = (fw - w[1:]).T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    return l2_tube_dyn


def get_rolling_l1_tube_dynamics(scaling, window_size):

    def l1_tube_dyn(z, v, w):
        l1 = scaling * ca.sum2(ca.fabs(v))
        fw = [ca.sum1(l1[max(i - window_size + 1, 0):i + 1]) / min(window_size, i + 1) for i in range(l1.numel())]
        g = ca.horzcat(*fw) - w[1:].T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    return l1_tube_dyn


def get_rolling_l2_tube_dynamics(scaling, window_size):

    def l2_tube_dyn(z, v, w):
        l2 = scaling * ca.sum2(v ** 2)
        fw = [ca.sum1(l2[max(i - window_size + 1, 0):i + 1]) / min(window_size, i + 1) for i in range(l2.numel())]
        g = ca.horzcat(*fw) - w[1:].T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    return l2_tube_dyn


def get_oneshot_nn_tube_dynamics(model_name, device='cuda'):
    api = wandb.Api()
    if device == 'cpu':
        model_cfg, state_dict = wandb_model_load_cpu(api, model_name)
    else:
        model_cfg, state_dict = wandb_model_load(api, model_name)

    H = model_cfg.dataset.H
    # TODO: proper sizing
    tube_oneshot_model = instantiate(model_cfg.model)(1 + 2 * H, H)

    tube_oneshot_model.load_state_dict(state_dict)

    tube_oneshot_model.to(device)
    tube_oneshot_model.eval()
    fw = l4c.L4CasADi(tube_oneshot_model, device=device)

    def oneshot_nn_tube_dyn(z, v, w):
        tube_input = ca.horzcat(w[0, :], z[0, 2:], ca.reshape(v, 1, v.numel()))
        g = fw(tube_input.T).T - w[1:].T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    return oneshot_nn_tube_dyn


def get_tube_dynamics(tube_dyn, scaling=0.5, window_size=10, nn_path=None, device='cuda'):
    if tube_dyn == 'l1':
        return get_l1_tube_dynamics(scaling)
    elif tube_dyn == 'l2':
        return get_l2_tube_dynamics(scaling)
    elif tube_dyn == 'l1_rolling':
        return get_rolling_l1_tube_dynamics(scaling, window_size)
    elif tube_dyn == 'l2_rolling':
        return get_rolling_l2_tube_dynamics(scaling, window_size)
    elif tube_dyn == 'NN_oneshot':
        return get_oneshot_nn_tube_dynamics(f'{nn_path}_model:best', device=device)
    else:
        raise ValueError(f'NN Tube dynamics {tube_dyn} not implemented')


class SolverCallback(ca.Callback):

    def __init__(self, name, debug_filename, x_cols, g_cols, p_cols, opts):
        ca.Callback.__init__(self)

        self.filename = debug_filename
        self.cols = ["iter"] + x_cols + g_cols
        self.g_cols = g_cols
        self.x_cols = x_cols
        self.p_cols = p_cols
        self.df = pd.DataFrame(columns=self.cols)
        self.it = 0

        self.nx = len(x_cols)
        self.ng = len(g_cols)
        self.np = len(p_cols)

        # Initialize internal objects
        self.construct(name, opts)

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_name_out(self, i):
        return "ret"

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n == 'f':
            return ca.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        else:
            return ca.Sparsity(0, 0)

    def eval(self, arg):
        # Create dictionary
        darg = {}
        for (i, s) in enumerate(ca.nlpsol_out()):
            darg[s] = arg[i]

        x = darg['x']
        g = darg['g']

        new_row_df = pd.DataFrame([np.concatenate((np.array([[self.it]]), np.array(x), np.array(g)), axis=0).squeeze()], columns=self.cols)
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)

        self.it += 1

        return [0]

    def write_data(self, solver, params):
        new_cols = ["lb_" + s for s in self.g_cols] + ["ub_" + s for s in self.g_cols] + ["lb_" + s for s in self.x_cols] + ["ub_" + s for s in self.x_cols]
        new_df = pd.DataFrame(np.zeros((self.df.shape[0], len(new_cols))), columns=new_cols)
        self.df = pd.concat([self.df, new_df], axis=1)
        self.df.loc[0, ["lb_" + s for s in self.g_cols]] = np.array(solver["lbg"]).squeeze()
        self.df.loc[0, ["ub_" + s for s in self.g_cols]] = np.array(solver["ubg"]).squeeze()
        self.df.loc[0, ["lb_" + s for s in self.x_cols]] = np.array(solver["lbx"]).squeeze()
        self.df.loc[0, ["ub_" + s for s in self.x_cols]] = np.array(solver["ubx"]).squeeze()
        self.df.loc[0, self.p_cols] = params.squeeze()

        self.df.to_csv(self.filename, index=False)
