import os
import torch
import wandb
import numpy as np
import pandas as pd
import casadi as ca
import l4casadi as l4c
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from deep_tube_learning.utils import wandb_model_load, wandb_model_load_cpu

Q = 10.
QW = 0.
R = 0.1
R_WARM = 1.
RV_FIRST = 10.
RV_SECOND = 10.
TWALL = 0.1
MPC_RECOMPUTE_DK = 1
problem_dict = {
    "gap": {"name": "gap", "start": np.array([0.0, 0.0]), "goal": np.array([1.5, 1.5]),
            "obs": {'cx': np.array([1., 0.]), 'cy': np.array([0.75, 1.5]), 'r': np.array([0.5, 0.5])},
            "Q": Q, "R_nominal": R, "R": R_WARM, "Qw": QW, "Rv_first": RV_FIRST, "Rv_second": RV_SECOND,
            "t_wall": TWALL, "mpc_dk": MPC_RECOMPUTE_DK},
    "right": {"name": "right", "start": np.array([0.0, 0.]), "goal": np.array([2., 0.]),
              "obs": {'cx': np.array([1., 1.]), 'cy': np.array([-0.425, 0.425]), 'r': np.array([0.375, 0.375])},
            "Q": Q, "R_nominal": R, "R": R_WARM, "Qw": QW, "Rv_first": RV_FIRST, "Rv_second": RV_SECOND,
            "t_wall": TWALL, "mpc_dk": MPC_RECOMPUTE_DK},
    "right_smol": {"name": "right", "start": np.array([0.0, 0.]), "goal": np.array([2., 0.]),
              "obs": {'cx': np.array([1., 1.]), 'cy': np.array([-0.4, 0.4]), 'r': np.array([0.375, 0.375])},
            "Q": Q, "R_nominal": R, "R": R_WARM, "Qw": QW, "Rv_first": RV_FIRST, "Rv_second": RV_SECOND,
            "t_wall": TWALL, "mpc_dk": MPC_RECOMPUTE_DK},
    "gap_big": {"name": "gap_big", "start": np.array([0., 0.]), "goal": np.array([3., 3.]),
            "obs": {'cx': np.array([2., 0.]), 'cy': np.array([1.75, 3.]), 'r': np.array([1., 1.])},
            "Q": Q, "R_nominal": R, "R": R_WARM, "Qw": QW, "Rv_first": RV_FIRST, "Rv_second": RV_SECOND,
            "t_wall": TWALL, "mpc_dk": MPC_RECOMPUTE_DK},
    "complex": {"name": "complex", "start": np.array([0.0, 0.]), "goal": np.array([2., 0.]),
            "obs": {'cx': np.array([0.5, 1.05, 1.65]), 'cy': np.array([-0.1, 0.2, -0.08]), 'r': np.array([0.25, 0.25, 0.2])},
            "Q": Q, "R_nominal": R, "R": R_WARM, "Qw": QW, "Rv_first": RV_FIRST, "Rv_second": RV_SECOND,
            "t_wall": TWALL, "mpc_dk": MPC_RECOMPUTE_DK},
    "none": {"name": "gap", "start": np.array([0.0, 0.0]), "goal": np.array([0.5, 0.]),
            "obs": {'cx': np.array([]), 'cy': np.array([]), 'r': np.array([])},
            "Q": Q, "R_nominal": R, "R": R_WARM, "Qw": QW, "Rv_first": RV_FIRST, "Rv_second": RV_SECOND,
            "t_wall": TWALL, "mpc_dk": MPC_RECOMPUTE_DK},
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
        if goal.shape == x.shape:
            dist = x - goal
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
    for k in range(1, z.shape[0]):
        if w is None:
            g_k, g_lb_k, g_ub_k = single_obstacle_constraint_k(z[k, :], obs_c, obs_r)
        else:
            g_k, g_lb_k, g_ub_k = single_obstacle_constraint_k(z[k, :], obs_c, obs_r + w[k - 1])
        g.append(g_k)
        g_lb.append(g_lb_k)
        g_ub.append(g_ub_k)
    return ca.horzcat(*g), ca.horzcat(*g_lb), ca.horzcat(*g_ub)


def obstacle_constraints(z, obs_c_x, obs_c_y, obs_r, w=None):
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
        g_i, g_lb_i, g_ub_i = single_obstacle_constraint(z, ca.horzcat(obs_c_x[i, :], obs_c_y[i, :]), obs_r[i, :], w=w)
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
    p_obs_c_x = ca.MX.sym("p_obs_c_x", Nobs, 1)  # positional obstacle centers
    p_obs_c_y = ca.MX.sym("p_obs_c_y", Nobs, 1)  # positional obstacle centers
    p_obs_r = ca.MX.sym("p_obs_r", Nobs, 1)  # positional obstacle radii

    # Make decision variables (2D double integrator)
    z = ca.MX.sym("z", N + 1, pm.n)
    v = ca.MX.sym("v", N, pm.m)

    # Make params for cost function
    p_z_cost = ca.MX.sym("p_z_cost", N + 1, pm.n)
    p_v_cost = ca.MX.sym("p_z_cost", N, pm.m)

    return z, v, z_lb, z_ub, v_lb, v_ub, p_z0, p_zf, p_z_cost, p_v_cost, p_obs_c_x, p_obs_c_y, p_obs_r


def trajopt_solver(pm, N, Q, R, Nobs, Qf=None, Rv_first=0, Rv_second=0, max_iter=1000, debug_filename=None, t_wall=None):
    z, v, z_lb, z_ub, v_lb, v_ub, p_z0, p_zf, p_z_cost, p_v_cost, p_obs_c_x, p_obs_c_y, p_obs_r = setup_trajopt_solver(pm, N, Nobs)

    if Qf is None:
        Qf = Q
    Q = ca.DM(Q)
    Qf = ca.DM(Qf)

    # Define NLP
    obj = quadratic_objective(z[:-1, :], Q, goal=p_z_cost[:-1, :]) + quadratic_objective(v, R, goal=p_v_cost) \
          + quadratic_objective(z[-1, :], Qf, goal=p_z_cost[-1, :])
    if Rv_first > 0:
        Rv_first = ca.DM(Rv_first)
        obj += quadratic_objective(v[:-1, :] - v[1:, :], Rv_first)
    if Rv_second > 0:
        Rv_second = ca.DM(Rv_second)
        first = v[:-1, :] - v[1:, :]
        obj += quadratic_objective(first[:-1, :] - first[1:, :], Rv_second)
    g_dyn, g_lb_dyn, g_ub_dyn = dynamics_constraint(pm.f, z, v)
    g_obs, g_lb_obs, g_ub_obs = obstacle_constraints(z, p_obs_c_x, p_obs_c_y, p_obs_r)
    g_ic, g_lb_ic, g_ub_ic = initial_condition_equality_constraint(z, p_z0)

    g = ca.horzcat(g_dyn, g_obs, g_ic)
    g_lb = ca.horzcat(g_lb_dyn, g_lb_obs, g_lb_ic)
    g_ub = ca.horzcat(g_ub_dyn, g_ub_obs, g_ub_ic)
    g = g.T
    g_lb = g_lb.T
    g_ub = g_ub.T

    # Generate solver
    x_nlp = ca.vertcat(
        ca.reshape(z, (N + 1) * pm.n, 1),
        ca.reshape(v, N * pm.m, 1),
    )
    lbx = ca.vertcat(
        ca.reshape(z_lb, (N + 1) * pm.n, 1),
        ca.reshape(v_lb, N * pm.m, 1),
    )
    ubx = ca.vertcat(
        ca.reshape(z_ub, (N + 1) * pm.n, 1),
        ca.reshape(v_ub, N * pm.m, 1),
    )
    p_nlp = ca.vertcat(
        p_z0.T, p_zf.T,
        ca.reshape(p_z_cost, (N + 1) * pm.n, 1), ca.reshape(p_v_cost, N * pm.m, 1),
        p_obs_c_x, p_obs_c_y, p_obs_r
    )

    x_cols, g_cols, p_cols = generate_col_names(pm, N, Nobs, x_nlp, g, p_nlp)
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
    else:
        nlp_opts['iteration_callback'] = None

    nlp_solver = ca.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx, "g_cols": g_cols, "x_cols": x_cols, "p_cols": p_cols}

    if t_wall is None:
        return solver, nlp_dict, nlp_opts

    t_lim_nlp_opts = nlp_opts.copy()
    t_lim_nlp_opts["ipopt.max_wall_time"] = t_wall
    t_lim_nlp_solver = ca.nlpsol("trajectory_generator", "ipopt", nlp_dict, t_lim_nlp_opts)

    t_lim_solver = {
        "solver": t_lim_nlp_solver, "callback": t_lim_nlp_opts['iteration_callback'],
        "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx,
        "g_cols": g_cols, "x_cols": x_cols, "p_cols": p_cols
    }
    return solver, nlp_dict, nlp_opts, t_lim_solver, t_lim_nlp_opts


def trajopt_tube_solver(pm, tube_dynamics, N, H_rev, Q, Qw, R, w_max, Nobs, Qf=None, Rv_first=0, Rv_second=0,
                        max_iter=1000, debug_filename=None, t_wall=None, solver_str="ipopt"):
    z, v, z_lb, z_ub, v_lb, v_ub, p_z0, p_zf, p_z_cost, p_v_cost, p_obs_c_x, p_obs_c_y, p_obs_r = setup_trajopt_solver(pm, N, Nobs)
    w = ca.MX.sym("w", N, 1)
    w_lb = ca.DM(N, 1)
    w_ub = ca.DM(np.ones((N, 1)) * w_max)
    e = ca.MX.sym("e", H_rev, 1)
    v_prev = ca.MX.sym("v_prev", H_rev, pm.m)

    if Qf is None:
        Qf = Q
    Q = ca.DM(Q)
    Qf = ca.DM(Qf)

    # Define NLP
    obj = quadratic_objective(z[:-1, :], Q, goal=p_z_cost[:-1, :]) + quadratic_objective(v, R, goal=p_v_cost) \
          + quadratic_objective(z[-1, :], Qf, goal=p_z_cost[-1, :]) + quadratic_objective(w, Qw)
    if Rv_first > 0:
        Rv_first = ca.DM(Rv_first)
        obj += quadratic_objective(v[:-1, :] - v[1:, :], Rv_first)
    if Rv_second > 0:
        Rv_second = ca.DM(Rv_second)
        first = v[:-1, :] - v[1:, :]
        obj += quadratic_objective(first[:-1, :] - first[1:, :], Rv_second)
    g_dyn, g_lb_dyn, g_ub_dyn = dynamics_constraint(pm.f, z, v)
    g_obs, g_lb_obs, g_ub_obs = obstacle_constraints(z, p_obs_c_x, p_obs_c_y, p_obs_r, w=w)
    g_ic, g_lb_ic, g_ub_ic = initial_condition_equality_constraint(z, p_z0)
    g_tube, g_lb_tube, g_ub_tube = tube_dynamics(z, v, w, e, v_prev)

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
    p_nlp = ca.vertcat(
        p_z0.T, p_zf.T,
        ca.reshape(p_z_cost, (N + 1) * pm.n, 1), ca.reshape(p_v_cost, N * pm.m, 1),
        p_obs_c_x, p_obs_c_y, p_obs_r,
        e, ca.reshape(v_prev, H_rev * pm.m, 1)
    )

    x_cols, g_cols, p_cols = generate_col_names(pm, N, Nobs, x_nlp, g, p_nlp, H_rev=H_rev)
    nlp_dict = {
        "x": x_nlp,
        "f": obj,
        "g": g,
        "p": p_nlp
    }
    if solver_str == "snopt":
        nlp_opts = {"snopt": {}}
    elif solver_str == "ipopt":
        nlp_opts = {
            "ipopt.linear_solver": "mumps",
            "ipopt.sb": "yes",
            "ipopt.max_iter": max_iter,
            "ipopt.tol": 1e-4,
            "print_time": True,
            # "ipopt.max_wall_time": 0.1,
            # "ipopt.constr_viol_tol": 1e-1,
            "ipopt.hessian_approximation": "limited-memory"  # Seems to help get constraint violation down, now mostly dual infeasibility remains
        }
    else:
        raise ValueError(f"solver {solver_str} not supported.")

    if debug_filename is not None and solver_str != "snopt":
        nlp_opts['iteration_callback'] = SolverCallback('iter_callback', x_cols, g_cols, p_cols, {})
    else:
        nlp_opts['iteration_callback'] = None

    nlp_solver = ca.nlpsol("trajectory_generator", solver_str, nlp_dict, nlp_opts)

    solver = {
        "solver": nlp_solver, "callback": nlp_opts['iteration_callback'],
        "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx,
        "g_cols": g_cols, "x_cols": x_cols, "p_cols": p_cols
    }

    if t_wall is None:
        return solver, nlp_dict, nlp_opts

    t_lim_nlp_opts = nlp_opts.copy()
    if solver_str == "snopt":
        t_lim_nlp_opts["snopt"]["Major feasibility tolerance"] = 1e-6
        t_lim_nlp_opts["snopt"]["Major optimality tolerance"] = 1e-3
        t_lim_nlp_opts["snopt"]["Major iterations limit"] = 4
        t_lim_nlp_opts["snopt"]["Minor iterations limit"] = 20
        t_lim_nlp_opts["snopt"]["Time limit"] = t_wall
    elif solver_str == "ipopt":
        t_lim_nlp_opts["ipopt.max_wall_time"] = t_wall
        t_lim_nlp_opts["ipopt.mu_init"] = 1e-3
        t_lim_nlp_opts["ipopt.barrier_tol_factor"] = 1e6
        t_lim_nlp_opts["ipopt.mu_strategy"] = "monotone"
        # t_lim_nlp_opts["ipopt.warm_start_init_point"] = 'yes'

    t_lim_nlp_solver = ca.nlpsol("trajectory_generator", solver_str, nlp_dict, t_lim_nlp_opts)
    # t_lim_nlp_solver.print_options()
    t_lim_solver = {
        "solver": t_lim_nlp_solver, "callback": t_lim_nlp_opts['iteration_callback'],
        "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx,
        "g_cols": g_cols, "x_cols": x_cols, "p_cols": p_cols
    }
    return solver, nlp_dict, nlp_opts, t_lim_solver, t_lim_nlp_opts


def generate_col_names(pm, N, Nobs, x, g, p, H_rev=0):
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
            np.reshape(z_str, ((N + 1) * pm.n, 1), order='F'),
            np.reshape(v_str, (N * pm.m, 1), order='F')
        )).squeeze())
    else:
        w_str = np.array(["w"] * N, dtype='U8').reshape((N, 1))
        for r in range(w_str.shape[0]):
            w_str[r, 0] = w_str[r, 0] + f"_{r}"
        x_cols = list(np.vstack((
            np.reshape(z_str, ((N + 1) * pm.n, 1), order='F'),
            np.reshape(v_str, (N * pm.m, 1), order='F'),
            w_str
        )).squeeze())

    g_dyn_cols = []
    for k in range(N):
        g_dyn_cols.extend(["dyn_" + st + f"_{k}" for st in pm.state_names])
    g_obs_cols = []
    for i in range(Nobs):
        g_obs_cols.extend([f"obs_{i}_{k}" for k in range(N)])
    g_tube_dyn = [f"tube_{k}" for k in range(N)]
    g_cols = g_dyn_cols + g_obs_cols + ["ic_" + st for st in pm.state_names]

    if not len(g_cols) == g.numel():
        g_cols.extend(g_tube_dyn)

    obs_c_lst = [f'obs_{i}_x' for i in range(Nobs)] + [f'obs_{i}_y' for i in range(Nobs)]
    if p.numel() == 2 * pm.n + len(obs_c_lst) + Nobs:
        p_cols = [f'z_ic_{i}' for i in range(pm.n)] + [f'z_g_{i}' for i in range(pm.n)] + obs_c_lst + \
                 [f'obs_{i}_r' for i in range(Nobs)]
    elif p.numel() == 2 * pm.n + len(obs_c_lst) + Nobs + H_rev + H_rev * pm.m:
        p_cols = [f'z_ic_{i}' for i in range(pm.n)] + [f'z_g_{i}' for i in range(pm.n)] + obs_c_lst + \
                 [f'obs_{i}_r' for i in range(Nobs)]
        e_cols = [f"e_{i}" for i in range(H_rev)]
        v_prev_str = np.array(["v_prev"] * (H_rev * pm.m), dtype='U13').reshape((H_rev, pm.m))
        for r in range(v_prev_str.shape[0]):
            for c in range(v_prev_str.shape[1]):
                v_prev_str[r, c] = v_prev_str[r, c] + f"_{r}_{c}"
        p_cols += e_cols + list(np.reshape(v_prev_str, (-1, 1), order='F').squeeze())
    else:
        z_cost = ["cost_" + s for s in np.reshape(z_str, ((N + 1) * pm.n,), order='F')]
        v_cost = ["cost_" + s for s in np.reshape(v_str, (N * pm.m,), order='F')]
        p_cols = [f'z_ic_{i}' for i in range(pm.n)] + [f'z_g_{i}' for i in range(pm.n)] + z_cost + v_cost + obs_c_lst + \
                 [f'obs_{i}_r' for i in range(Nobs)]
        e_cols = [f"e_{i}" for i in range(H_rev)]
        v_prev_str = np.array(["v_prev"] * (H_rev * pm.m), dtype='U13').reshape((H_rev, pm.m))
        for r in range(v_prev_str.shape[0]):
            for c in range(v_prev_str.shape[1]):
                v_prev_str[r, c] = v_prev_str[r, c] + f"_{r}_{c}"
        p_cols += e_cols + list(np.reshape(v_prev_str, (-1, 1)).squeeze())

    assert len(x_cols) == x.numel() and len(g_cols) == g.numel() and len(p_cols) == p.numel()
    return x_cols, g_cols, p_cols


def init_params(z0, zf, obs, z_cost=None, v_cost=None, e=None, v_prev=None):
    Nobs = len(obs['r'])
    if z_cost is None:
        params = np.vstack([z0[:, None], zf[:, None], obs['cx'][:, None], obs['cy'][:, None], obs['r'][:, None]])
    else:
        params = np.vstack([
            z0[:, None], zf[:, None],
            np.reshape(z_cost, (-1, 1), order='F'), np.reshape(v_cost, (-1, 1), order='F'),
            obs['cx'][:, None], obs['cy'][:, None], obs['r'][:, None]
        ])
    if e is not None:
        params = np.vstack([params, e, np.reshape(v_prev, (-1, 1), order='F')])
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
            np.reshape(w, (N, 1), order='F')
        ])
    return x_init


def extract_solution(sol, N, n, m):
    z_ind = (N + 1) * n
    v_ind = N * m
    z_sol = np.array(sol["x"][:z_ind, :].reshape((N + 1, n)))
    v_sol = np.array(sol["x"][z_ind:z_ind + v_ind, :].reshape((N, m)))
    if sol["x"].numel() > z_ind + v_ind:
        w_sol = np.array(sol["x"][z_ind + v_ind:, :].reshape((N, 1)))
        return z_sol, v_sol, w_sol
    else:
        return z_sol, v_sol


def plot_problem(ax, obs, z0, zf):
    for i in range(len(obs["r"])):
        xc = obs['cx'][i]
        yc = obs['cy'][i]
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


def get_warm_start(warm_start, start, goal, N, planning_model, obs=None, Q=None, R=None,
                   Rv_first=0, Rv_second=0, nominal_ws='interpolate'):
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
        sol, solver = solve_nominal(start, goal, obs, planning_model, N, Q, R, Rv_first=Rv_first, Rv_second=Rv_second, warm_start=nominal_ws)
        z_init, v_init = extract_solution(sol, N, planning_model.n, planning_model.m)
    else:
        raise ValueError(f'Warm start {warm_start} not implemented. Must be ic, goal, interpolate, or nominal')

    return z_init, v_init


def get_tube_warm_start(w_init, tube_dynamics, z, v, e, v_prev):
    if w_init == "evaluate":
        return tube_dynamics(z, v, e, v_prev).cpu().numpy()
    elif isinstance(w_init, (int, float)):
        return np.ones((z.shape[0] - 1, 1)) * w_init
    raise ValueError(f"Tube warm start {w_init} not implemented. Must be evaluate or a double")


def solve_nominal(start, goal, obs, planning_model, N, Q, R, Rv_first=0, Rv_second=0, warm_start='start', debug_filename=None):
    solver, nlp_dict, nlp_opts = trajopt_solver(planning_model, N, Q, R, len(obs["r"]),
                                                Rv_first=Rv_first, Rv_second=Rv_second, debug_filename=debug_filename)

    z_init, v_init = get_warm_start(warm_start, start, goal, N, planning_model)

    params = init_params(start, goal, obs)
    x_init = init_decision_var(z_init, v_init)

    sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"],
                           ubx=solver["ubx"])

    if nlp_opts['iteration_callback'] is not None:
        nlp_opts['iteration_callback'].write_data(solver, params)
    return sol, solver


def solve_tube(
        start, goal, obs, planning_model, tube_dynamics, eval_tube, N, H_rev, Q, Qw, R, w_max, Qf=None, R_nominal=None,
        Rv_first=0, Rv_second=0, warm_start='start', nominal_ws='interpolate', tube_ws=0, debug_filename=None,
        max_iter=1000, track_warm=False, solver_str="ipopt"
):
    if R_nominal is None:
        R_nominal = R
    z_init, v_init = get_warm_start(warm_start, start, goal, N, planning_model, obs, Q, R_nominal,
                                    Rv_first=Rv_first, Rv_second=Rv_second, nominal_ws=nominal_ws)
    e = np.zeros((H_rev, 1))
    v_prev = np.zeros((H_rev, planning_model.m))

    solver, nlp_dict, nlp_opts = trajopt_tube_solver(
        planning_model, tube_dynamics, N, H_rev, Q, Qw, R, w_max, len(obs['r']), Qf=Qf,
        Rv_first=Rv_first, Rv_second=Rv_second, max_iter=max_iter, debug_filename=debug_filename, solver_str=solver_str
    )

    w_init = get_tube_warm_start(tube_ws, eval_tube, z_init, v_init, e, v_prev)

    if track_warm:
        z_cost = z_init.copy()
        v_cost = v_init.copy()
    else:
        z_cost = np.repeat(goal[None, :], N + 1, axis=0)
        v_cost = np.zeros((N, planning_model.m))
    params = init_params(start, goal, obs, z_cost=z_cost, v_cost=v_cost, e=e, v_prev=v_prev)
    x_init = init_decision_var(z_init, v_init, w=w_init)

    sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"],
                           ubx=solver["ubx"])

    if nlp_opts['iteration_callback'] is not None:
        nlp_opts['iteration_callback'].write_data(solver, params, debug_filename)
    return sol, solver


def get_l1_tube_dynamics(scaling):

    def l1_tube_dyn(z, v, w, e, v_prev):
        fw = scaling * ca.sum2(ca.fabs(v))
        g = (fw - w).T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    return l1_tube_dyn


def get_l2_tube_dynamics(scaling):

    def l2_tube_dyn(z, v, w, e, v_prev):
        fw = scaling * ca.sum2(v ** 2)
        g = (fw - w).T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    return l2_tube_dyn


def get_rolling_l1_tube_dynamics(scaling, window_size):

    def l1_tube_dyn(z, v, w, e, v_prev):
        l1 = scaling * ca.sum2(ca.fabs(v))
        fw = [ca.sum1(l1[max(i - window_size + 1, 0):i + 1]) / min(window_size, i + 1) for i in range(l1.numel())]
        g = ca.horzcat(*fw) - w.T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    return l1_tube_dyn


def get_rolling_l2_tube_dynamics(scaling, window_size):

    def l2_tube_dyn(z, v, w, e, v_prev):
        l2 = scaling * ca.sum2(v ** 2)
        fw = [ca.sum1(l2[max(i - window_size + 1, 0):i + 1]) / min(window_size, i + 1) for i in range(l2.numel())]
        g = ca.horzcat(*fw) - w.T
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

    H_fwd = model_cfg.dataset.H_fwd
    H_rev = model_cfg.dataset.H_rev
    # TODO: proper sizing
    tube_oneshot_model = instantiate(model_cfg.model)(H_rev + 2 * (H_rev + H_fwd), H_fwd)

    tube_oneshot_model.load_state_dict(state_dict)

    tube_oneshot_model.to(device)
    tube_oneshot_model.eval()
    fw = l4c.L4CasADi(tube_oneshot_model, device=device)

    def oneshot_nn_tube_dyn(z, v, w, e, v_prev):
        v_total = ca.vertcat(v_prev, v)
        tube_input = ca.horzcat(e.T, z[0, 2:], ca.reshape(v_total.T, 1, v_total.numel()))  # Note this transpose to get ca.reshape and np.reshape to agree
        print(tube_input, fw(tube_input.T).T, w, "\n")
        g = fw(tube_input.T).T - w.T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    def eval_oneshot_nn_tube_dyn(z, v, e, v_prev):
        v = torch.from_numpy(v).float().to(device)
        e = torch.from_numpy(e).float().to(device)
        v_prev = torch.from_numpy(v_prev).float().to(device)
        v = torch.concatenate((v_prev, v))
        v_total = torch.concatenate((v_prev, v))
        tube_input = ca.horzcat(e.T, torch.reshape(v_total, (1, v_total.numel())))  # Note this transpose to get ca.reshape and np.reshape to agree
        return tube_oneshot_model(tube_input)

    return oneshot_nn_tube_dyn, H_fwd, H_rev, eval_oneshot_nn_tube_dyn

def get_recursive_nn_tube_dynamics(model_name, device='cuda'):
    api = wandb.Api()
    if device == 'cpu':
        model_cfg, state_dict = wandb_model_load_cpu(api, model_name)
    else:
        model_cfg, state_dict = wandb_model_load(api, model_name)

    H_fwd = model_cfg.dataset.H_fwd
    H_rev = model_cfg.dataset.H_rev
    # TODO: proper sizing
    tube_recursive_model = instantiate(model_cfg.model)(H_rev + 2 * (H_rev + H_fwd), H_fwd)

    tube_recursive_model.load_state_dict(state_dict)

    tube_recursive_model.mlp.to(device)
    tube_recursive_model.mlp.eval()
    fw = l4c.L4CasADi(tube_recursive_model.mlp, device=device)

    def recursive_nn_tube_dyn(z, v, w, e, v_prev):
        v = ca.vertcat(v_prev, v)
        all_data = []
        for i in range(w.shape[0]):
            if i < tube_recursive_model.H_rev:
                data = ca.horzcat(
                    e[i:, :].T, w[0:i, :].T,
                    ca.reshape(v[i:i + tube_recursive_model.H_rev, :].T, 1, tube_recursive_model.H_rev * v.shape[1]),
                    ca.DM([i])
                )
            else:
                data = ca.horzcat(
                    w[i - tube_recursive_model.H_rev:i, :].T,
                    ca.reshape(v[i:i + tube_recursive_model.H_rev, :].T, 1, tube_recursive_model.H_rev * v.shape[1]),
                    ca.DM([i])
                )
            all_data.append(data)

        g = ca.horzcat(*[fw(data.T) for data in all_data]) - w.T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    def eval_nn_tube_dyn(z, v, e, v_prev):
        w = torch.zeros((v.shape[0], 1), device=device)
        v = torch.from_numpy(v).float().to(device)
        e = torch.from_numpy(e).float().to(device)
        v_prev = torch.from_numpy(v_prev).float().to(device)
        v = torch.concatenate((v_prev, v))
        for i in range(w.shape[0]):
            if i < tube_recursive_model.H_rev:
                data = torch.concatenate((
                    e[i:, :].T, w[0:i, :].T,
                    torch.reshape(v[i:i + tube_recursive_model.H_rev, :], (1, tube_recursive_model.H_rev * v.shape[1])),
                    torch.tensor([[i]], device=device).float()
                ), dim=1)
            else:
                data = torch.concatenate((
                    w[i - tube_recursive_model.H_rev:i, :].T,
                    torch.reshape(v[i:i + tube_recursive_model.H_rev, :], (1, tube_recursive_model.H_rev * v.shape[1])),
                    torch.tensor([[i]], device=device)
                ), dim=1)
            w[i] = tube_recursive_model.mlp(data)

        return w

    return recursive_nn_tube_dyn, H_fwd, H_rev, eval_nn_tube_dyn


def get_oneshot_nn_tube_dynamics_v2(model_name, device='cuda'):
    api = wandb.Api()
    if device == 'cpu':
        model_cfg, state_dict = wandb_model_load_cpu(api, model_name)
    else:
        model_cfg, state_dict = wandb_model_load(api, model_name)

    H_fwd = model_cfg.dataset.H_fwd
    H_rev = model_cfg.dataset.H_rev
    # TODO: proper sizing
    tube_oneshot_model = instantiate(model_cfg.model)(H_rev + 2 * (H_rev + H_fwd), H_fwd)

    tube_oneshot_model.load_state_dict(state_dict)

    tube_oneshot_model.to(device)
    tube_oneshot_model.eval()
    fw = l4c.L4CasADi(tube_oneshot_model, device=device)

    def oneshot_nn_tube_dyn(z, v, w, e, v_prev):
        v_total = ca.vertcat(v_prev, v)
        tube_input = ca.horzcat(e.T, z[0, 2:], ca.reshape(v_total.T, 1, v_total.numel()))  # Note this transpose to get ca.reshape and np.reshape to agree
        print(tube_input, fw(tube_input.T).T, w, "\n")
        g = fw(tube_input) - w.T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    def eval_oneshot_nn_tube_dyn(z, v, e, v_prev):
        v = torch.from_numpy(v).float().to(device)
        e = torch.from_numpy(e).float().to(device)
        v_prev = torch.from_numpy(v_prev).float().to(device)
        v = torch.concatenate((v_prev, v))
        v_total = torch.concatenate((v_prev, v))
        tube_input = ca.horzcat(e.T, torch.reshape(v_total, (1, v_total.numel())))  # Note this transpose to get ca.reshape and np.reshape to agree
        return tube_oneshot_model(tube_input)

    return oneshot_nn_tube_dyn, H_fwd, H_rev, eval_oneshot_nn_tube_dyn

def get_recursive_nn_tube_dynamics_v2(model_name, device='cuda'):
    api = wandb.Api()
    if device == 'cpu':
        model_cfg, state_dict = wandb_model_load_cpu(api, model_name)
    else:
        model_cfg, state_dict = wandb_model_load(api, model_name)

    H_fwd = model_cfg.dataset.H_fwd
    H_rev = model_cfg.dataset.H_rev
    # TODO: proper sizing
    tube_recursive_model = instantiate(model_cfg.model)(H_rev + 2 * (H_rev + H_fwd), H_fwd)

    tube_recursive_model.load_state_dict(state_dict)

    tube_recursive_model.mlp.to(device)
    tube_recursive_model.mlp.eval()
    fw = l4c.L4CasADi(tube_recursive_model.mlp, device=device)

    def recursive_nn_tube_dyn(z, v, w, e, v_prev):
        v = ca.vertcat(v_prev, v)
        all_data = []
        for i in range(w.shape[0]):
            if i < tube_recursive_model.H_rev:
                data = ca.horzcat(
                    e[i:, :].T, w[0:i, :].T,
                    ca.reshape(v[i:i + tube_recursive_model.H_rev, :].T, 1, tube_recursive_model.H_rev * v.shape[1]),
                    ca.DM([i])
                )
            else:
                data = ca.horzcat(
                    w[i - tube_recursive_model.H_rev:i, :].T,
                    ca.reshape(v[i:i + tube_recursive_model.H_rev, :].T, 1, tube_recursive_model.H_rev * v.shape[1]),
                    ca.DM([i])
                )
            all_data.append(data)

        g = fw(ca.vertcat(*all_data)).T - w.T
        g_lb = ca.DM(*g.shape)
        g_ub = ca.DM(*g.shape)

        return g, g_lb, g_ub

    def eval_nn_tube_dyn(z, v, e, v_prev):
        w = torch.zeros((v.shape[0],), device=device)
        v = torch.from_numpy(v).float().to(device)
        e = torch.from_numpy(e).float().to(device)
        v_prev = torch.from_numpy(v_prev).float().to(device)
        v = torch.concatenate((v_prev, v))
        for i in range(w.shape[0]):
            if i < tube_recursive_model.H_rev:
                data = torch.concatenate((
                    e[i:, :].T, w[0:i, :].T,
                    torch.reshape(v[i:i + tube_recursive_model.H_rev, :], (1, tube_recursive_model.H_rev * v.shape[1])),
                    torch.tensor([i], device=device).float()
                ), dim=1)
            else:
                data = torch.concatenate((
                    w[i - tube_recursive_model.H_rev:i, :].T,
                    torch.reshape(v[i:i + tube_recursive_model.H_rev, :], (1, tube_recursive_model.H_rev * v.shape[1])),
                    torch.tensor([i], device=device)
                ), dim=1)
            w[i] = tube_recursive_model.mlp(data)

        return w

    return recursive_nn_tube_dyn, H_fwd, H_rev, eval_nn_tube_dyn


def get_tube_dynamics(tube_dyn, scaling=0.5, window_size=10, nn_path=None, device='cuda'):
    if tube_dyn == 'l1':
        return get_l1_tube_dynamics(scaling), 0
    elif tube_dyn == 'l2':
        return get_l2_tube_dynamics(scaling), 0
    elif tube_dyn == 'l1_rolling':
        return get_rolling_l1_tube_dynamics(scaling, window_size), window_size
    elif tube_dyn == 'l2_rolling':
        return get_rolling_l2_tube_dynamics(scaling, window_size), window_size
    elif tube_dyn == 'NN_oneshot':
        if 'v2' in os.environ.get('CONDA_PREFIX'):
            return get_oneshot_nn_tube_dynamics_v2(f'{nn_path}_model:best', device=device)
        else:
            return get_oneshot_nn_tube_dynamics(f'{nn_path}_model:best', device=device)
    elif tube_dyn == 'NN_recursive':
        if 'v2' in os.environ.get('CONDA_PREFIX'):
            return get_recursive_nn_tube_dynamics_v2(f'{nn_path}_model:best', device=device)
        else:
            return get_recursive_nn_tube_dynamics(f'{nn_path}_model:best', device=device)
    else:
        raise ValueError(f'NN Tube dynamics {tube_dyn} not implemented')


class SolverCallback(ca.Callback):

    def __init__(self, name, x_cols, g_cols, p_cols, opts):
        ca.Callback.__init__(self)

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

    def write_data(self, solver, params, filename):
        new_cols = ["lb_" + s for s in self.g_cols] + ["ub_" + s for s in self.g_cols] + ["lb_" + s for s in self.x_cols] + ["ub_" + s for s in self.x_cols]
        new_df = pd.DataFrame(np.zeros((self.df.shape[0], len(new_cols))), columns=new_cols)
        self.df = pd.concat([self.df, new_df], axis=1)
        self.df.loc[0, ["lb_" + s for s in self.g_cols]] = np.array(solver["lbg"]).squeeze()
        self.df.loc[0, ["ub_" + s for s in self.g_cols]] = np.array(solver["ubg"]).squeeze()
        self.df.loc[0, ["lb_" + s for s in self.x_cols]] = np.array(solver["lbx"]).squeeze()
        self.df.loc[0, ["ub_" + s for s in self.x_cols]] = np.array(solver["ubx"]).squeeze()
        self.df.loc[0, self.p_cols] = params.squeeze()

        self.df.to_csv(filename, index=False)
        self.df = pd.DataFrame(columns=self.cols)
        self.it = 0
