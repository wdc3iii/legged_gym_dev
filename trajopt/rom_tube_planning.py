import time
import numpy as np
import casadi as ca
import l4casadi as l4c
import matplotlib.pyplot as plt

from trajopt.rom_dynamics import (SingleInt2D, DoubleInt2D, Unicycle, LateralUnicycle,
                                  ExtendedUnicycle, ExtendedLateralUnicycle)

# model = "SingleInt2D"
# model = "DoubleInt2D"
# model = "Unicycle"
# model = "LateralUnicycle"
# model = "ExtendedUnicycle"
model = "ExtendedLateralUnicycle"

start = np.array([-5, -5, 0])
goal = np.array([8, 3, np.pi / 2])

acc_max = 2
alpha_max = 4
vel_max = 1
omega_max = 2
pos_max = 10
dt = 0.25
N = 75

# obs = {
#     'c': np.array([[-2, 4, 1, 0], [-3, 5, -4, 0]]),
#     'r': np.array([1, 1, 0.2, 3])
# }
obs = {
    'c': np.array([[0, 2, 6, 4], [0, -4.5, 1, 3.2]]),
    'r': np.array([3, 1, 1, 1.5])
}
# obs = {
#     'c': np.zeros((0, 2)),
#     'r': np.zeros((0,))
# }


# TODO: error dynamics + tube solver
def trajopt_error_tube_solver(pm, N, Q, R, Nobs, Qf=None):
    raise NotImplementedError


def add_obs(pm, k, z, w, p_obs_c, p_obs_r, tube_dim, Nobs, g_lb, g_ub):
    for i in range(Nobs):
        if tube_dim == 1:
            # Simply increase distance from center by tube dimension
            g = ca.horzcat(g, ca.sum2((z[k, :2] - p_obs_c[i, :]) ** 2) - (p_obs_r[i, :] + w[k, :]) ** 2)
            g_lb = ca.horzcat(g_lb, ca.DM([0]))
            g_ub = ca.horzcat(g_ub, ca.DM.inf())
        elif tube_dim == pm.n:
            ellipse = (z[k, 0] - p_obs_c[i, 0]) ** 2 / (2 * (w[k, 0] + p_obs_r[i]) ** 2) + \
                      (z[k, 1] - p_obs_c[i, 1]) ** 2 / (2 * (w[k, 0] + p_obs_r[i]) ** 2)
            g = ca.horzcat(g, ellipse - 1)
            g_lb = ca.horzcat(g_lb, ca.DM([0]))
            g_ub = ca.horzcat(g_ub, ca.DM.inf())
        else:
            raise ValueError(f"Tube Dimension {tube_dim} not supported for obstacles")
    return g, g_lb, g_ub


# TODO: tube solver
def trajopt_tube_solver(pm, tube_dyn_model, tube_dim, w_max, N, Q, R, Nobs, Qf=None, device='cpu'):
    if Qf is None:
        Qf = Q
    Q = ca.DM(Q)
    Qf = ca.DM(Qf)
    z_min = ca.DM(pm.z_min)
    z_max = ca.DM(pm.z_max)
    v_min = ca.DM(pm.v_min)
    v_max = ca.DM(pm.v_max)

    fw = l4c.L4CasADi(tube_dyn_model, device=device)

    # Make decision variables (2D double integrator)
    z = ca.MX.sym("z", N + 1, pm.n)
    v = ca.MX.sym("v", N, pm.m)
    w = ca.MX.sym("w", N + 1, tube_dim)

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
    w_lb = ca.DM(N + 1, tube_dim)
    w_ub = ca.DM(np.ones((N + 1, tube_dim)) * w_max)

    for k in range(N):
        # cost function
        obj += (z[k, :] - p_zf) @ Q @ (z[k, :] - p_zf).T + v[k, :] @ R @ v[k, :].T

        # dynamics
        g = ca.horzcat(g, pm.f(z[k, :].T, v[k, :].T).T - z[k + 1, :])
        g_lb = ca.horzcat(g_lb, ca.DM(np.zeros((pm.n,))).T)
        g_ub = ca.horzcat(g_ub, ca.DM(np.zeros((pm.n,))).T)

        # Tube dynamics
        tube_input = ca.horzcat(w[k, :], z[k, :], v[k, :])
        g = ca.horzcat(g, fw(tube_input) - w[k + 1, :])
        g_lb = ca.horzcat(g_lb, ca.DM(np.zeros((tube_dim,))).T)
        g_ub = ca.horzcat(g_ub, ca.DM(np.zeros((tube_dim,))).T)

        # obstacle constraints
        # TODO: check outer ellipse approximation for vector tube based obstacle constraints
        # for i in range(Nobs):
        #     if tube_dim == 1:
        #         # Simply increase distance from center by tube dimension
        #         g = ca.horzcat(g, ca.sum2((z[k, :2] - p_obs_c[i, :]) ** 2) - (p_obs_r[i, :] + w[k, :]) ** 2)
        #         g_lb = ca.horzcat(g_lb, ca.DM([0]))
        #         g_ub = ca.horzcat(g_ub, ca.DM.inf())
        #     elif tube_dim == pm.n:
        #         ellipse = (z[k, 0] - p_obs_c[i, 0]) ** 2 / (2 * (w[k, 0] + p_obs_r[i])**2) + \
        #                   (z[k, 1] - p_obs_c[i, 1]) ** 2 / (2 * (w[k, 0] + p_obs_r[i])**2)
        #         g = ca.horzcat(g, ellipse - 1)
        #         g_lb = ca.horzcat(g_lb, ca.DM([0]))
        #         g_ub = ca.horzcat(g_ub, ca.DM.inf())
        #     else:
        #         raise ValueError(f"Tube Dimension {tube_dim} not supported for obstacles")
        g, g_lb, g_ub = add_obs(pm, k, z, w, p_obs_c, p_obs_r, tube_dim, Nobs, g_lb, g_ub)

    # Terminal cost/constraints
    obj += (z[N, :] - p_zf) @ Qf @ (z[N, :] - p_zf).T
    # for i in range(Nobs):
    #     if tube_dim == 1:
    #         # Simply increase distance from center by tube dimension
    #         g = ca.horzcat(g, ca.sum2((z[N, :2] - p_obs_c[i, :]) ** 2) - (p_obs_r[i, :] + w[N, :]) ** 2)
    #         g_lb = ca.horzcat(g_lb, ca.DM([0]))
    #         g_ub = ca.horzcat(g_ub, ca.DM.inf())
    #     elif tube_dim == pm.n:
    #         ellipse = (z[N, 0] - p_obs_c[i, 0]) ** 2 / (2 * (w[N, 0] + p_obs_r[i]) ** 2) + \
    #                   (z[N, 1] - p_obs_c[i, 1]) ** 2 / (2 * (w[N, 0] + p_obs_r[i]) ** 2)
    #         g = ca.horzcat(g, ellipse - 1)
    #         g_lb = ca.horzcat(g_lb, ca.DM([0]))
    #         g_ub = ca.horzcat(g_ub, ca.DM.inf())
    #     else:
    #         raise ValueError(f"Tube Dimension {tube_dim} not supported for obstacles")
    g, g_lb, g_ub = add_obs(pm, N, z, w, p_obs_c, p_obs_r, tube_dim, Nobs, g_lb, g_ub)

    # Initial condition
    if tube_dim == 1:
        g = ca.horzcat(g, w[0, :] - ca.dot(z[0, :] - p_z0, z[0, :] - p_z0))
    elif tube_dim == pm.n:
        g = ca.horzcat(g, w[0, :] - ca.fabs(z[0, :] - p_z0))
    else:
        raise ValueError(f"Tube Dimension {tube_dim} not supported for initial condition")
    g_lb = ca.horzcat(g_lb, ca.DM(np.zeros((tube_dim,))).T)
    g_ub = ca.horzcat(g_ub, ca.DM(np.zeros((tube_dim,))).T)

    # Generate solver
    x_nlp = ca.vertcat(
        ca.reshape(z, (N + 1) * pm.n, 1),
        ca.reshape(v, N * pm.m, 1),
        ca.reshape(w, (N + 1) * tube_dim, 1)
    )
    lbx = ca.vertcat(
        ca.reshape(z_lb, (N + 1) * pm.n, 1),
        ca.reshape(v_lb, N * pm.m, 1),
        ca.reshape(w_lb, (N + 1) * pm.n, 1)
    )
    ubx = ca.vertcat(
        ca.reshape(z_ub, (N + 1) * pm.n, 1),
        ca.reshape(v_ub, N * pm.m, 1),
        ca.reshape(w_ub, (N + 1) * tube_dim, 1)
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
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-4,
        # "ipopt.print_level": 5,
        "print_time": True,
    }

    nlp_solver = ca.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx}

    return solver


def generate_trajectory(plan_model, z0, zf, tube_dyn_model, tube_dim, w_max, N, Q, R, Qf=None):
    Nobs = len(obs['r'])

    nlp = trajopt_tube_solver(plan_model, tube_dyn_model, tube_dim, w_max, N, Q, R, Nobs, Qf=Qf)

    params = np.vstack([z0[:, None], zf[:, None], np.reshape(obs['c'], (2 * Nobs, 1)), obs['r'][:, None]])

    v_init = np.zeros((N, plan_model.m))
    # z_init = np.repeat(xf[:, None], N + 1, 1)
    # z_init = np.repeat(x0[:, None], N + 1, 1)
    z_init = np.outer(np.linspace(0, 1, N+1), (zf - z0)) + z0
    w_init = np.zeros((N + 1, tube_dim))

    x_init = np.vstack([
        np.reshape(z_init, ((N + 1) * plan_model.n, 1)),
        np.reshape(v_init, (N * plan_model.m, 1)),
        np.reshape(w_init, ((N + 1) * tube_dim, 1))
    ])

    tic = time.perf_counter_ns()
    sol = nlp["solver"](x0=x_init, p=params, lbg=nlp["lbg"], ubg=nlp["ubg"], lbx=nlp["lbx"], ubx=nlp["ubx"])
    toc = time.perf_counter_ns()
    print(f"Solve Time: {(toc - tic) / 1e6}ms")

    # extract solution
    z_ind = (N + 1) * plan_model.n
    v_ind = N * plan_model.m
    z_sol = np.array(sol["x"][:z_ind, :].reshape((N + 1, plan_model.n)))
    v_sol = np.array(sol["x"][z_ind:z_ind + v_ind, :].reshape((N, plan_model.m)))
    w_sol = np.array(sol["x"][z_ind + v_ind:, :].reshape((N + 1, tube_dim)))

    fig, axs = plt.subplots(2,1)
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
    plan_model.plot_spacial(ax, z_sol)
    plt.axis("square")
    plt.show()

    print("Complete")


if __name__ == "__main__":
    if model == "SingleInt2D":
        z_max = np.array([pos_max, pos_max])
        v_max = np.array([vel_max, vel_max])
        planning_model = SingleInt2D(dt, -z_max, z_max, -v_max, v_max)

        Q = 10 * np.eye(2)
        R = 0.1 * np.eye(2)
        z0 = start[:2]
        zf = goal[:2]

        generate_trajectory(planning_model, z0, zf, N, Q, R)

    elif model == "DoubleInt2D":
        z_max = np.array([pos_max, pos_max, vel_max, vel_max])
        v_max = np.array([acc_max, acc_max])
        planning_model = DoubleInt2D(dt, -z_max, z_max, -v_max, v_max)

        Q = np.diag([10, 10, 5, 5])
        R = 0.1 * np.eye(2)
        z0 = np.hstack([start[:2], np.zeros((2,))])
        zf = np.hstack([goal[:2], np.zeros((2,))])

        generate_trajectory(planning_model, z0, zf, N, Q, R)

    elif model == "Unicycle":
        z_max = np.array([pos_max, pos_max, np.inf])
        v_max = np.array([vel_max, omega_max])
        v_min = np.array([-vel_max / 2, -omega_max])
        planning_model = Unicycle(dt, -z_max, z_max, v_min, v_max)

        Q = np.diag([10, 10, 0])
        Qf = np.diag([100, 100, 1])
        R = 0.1 * np.eye(2)
        z0 = start
        zf = goal
        generate_trajectory(planning_model, z0, zf, N, Q, R, Qf=Qf)

    elif model == "LateralUnicycle":
        z_max = np.array([pos_max, pos_max, np.inf])
        v_max = np.array([vel_max, vel_max / 2, omega_max])
        v_min = np.array([-vel_max / 2, -vel_max / 2, -omega_max])
        planning_model = LateralUnicycle(dt, -z_max, z_max, v_min, v_max)

        Q = np.diag([10, 10, 0])
        Qf = np.diag([100, 100, 1])
        R = np.diag([0.1, 1, 0.1])
        z0 = start
        zf = goal

        generate_trajectory(planning_model, z0, zf, N, Q, R, Qf=Qf)

    elif model == "ExtendedUnicycle":
        z_max = np.array([pos_max, pos_max, np.inf, vel_max, omega_max])
        z_min = -np.array([pos_max, pos_max, np.inf, vel_max / 2, omega_max])
        v_max = np.array([acc_max, alpha_max])
        planning_model = ExtendedUnicycle(dt, z_min, z_max, -v_max, v_max)

        Q = np.diag([10, 10, 0, 5, 5])
        Qf = np.diag([100, 100, 1, 100, 100])
        R = 0.1 * np.eye(2)
        z0 = np.hstack([start, np.zeros((2,))])
        zf = np.hstack([goal, np.zeros((2,))])
        generate_trajectory(planning_model, z0, zf, N, Q, R, Qf=Qf)

    elif model == "ExtendedLateralUnicycle":
        z_max = np.array([pos_max, pos_max, np.inf, vel_max, vel_max / 2, omega_max])
        z_min = -np.array([pos_max, pos_max, np.inf, vel_max / 2, vel_max / 2, omega_max])
        v_max = np.array([acc_max, acc_max / 2, alpha_max])
        planning_model = ExtendedLateralUnicycle(dt, z_min, z_max, -v_max, v_max)

        Q = np.diag([10, 10, 0, 5, 5, 5])
        Qf = np.diag([100, 100, 1, 100, 100, 100])
        R = 0.1 * np.eye(3)
        z0 = np.hstack([start, np.zeros((3,))])
        zf = np.hstack([goal, np.zeros((3,))])
        generate_trajectory(planning_model, z0, zf, N, Q, R, Qf=Qf)
