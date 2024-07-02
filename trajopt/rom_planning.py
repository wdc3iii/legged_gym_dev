import time
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt


x0 = np.array([-5, -5, 0, 0])
xf = np.array([8, 3, 0, 0])
Q = np.eye(4)
R = np.eye(2)
u_max = 2
v_max = 1
p_max = 10
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

def trajopt_solver(N, dt, Q, R, v_max, p_max, u_max, Nobs, Qf=None):
    if Qf is None:
        Qf = Q
    Q = cs.DM(Q)
    Qf = cs.DM(Qf)
    v_max = cs.DM(v_max)
    p_max = cs.DM(p_max)
    u_max = cs.DM(u_max)

    # Make decision variables (2D double integrator)
    x = cs.MX.sym("x", N + 1, 4)
    u = cs.MX.sym("u", N, 2)

    # Parameters: initial condition, final condition
    p_x0 = cs.MX.sym("p_x0", 1, 4)                 # Initial condition
    p_xf = cs.MX.sym("p_xf", 1, 4)                 # Goal state
    p_obs_c = cs.MX.sym("p_obs_c", Nobs, 2)    # obstacle centers
    p_obs_r = cs.MX.sym("p_obs_r", Nobs, 1)    # obstacle radii

    # Define NLP
    obj = 0
    g = []
    g_lb = []
    g_ub = []

    state_bound = cs.DM([p_max, p_max, v_max, v_max]).T
    input_bound = cs.DM([u_max, u_max]).T

    x_lb = cs.repmat(-state_bound, N + 1, 1)
    x_ub = cs.repmat(state_bound, N + 1, 1)
    u_lb = cs.repmat(-input_bound, N, 1)
    u_ub = cs.repmat(input_bound, N, 1)

    A = cs.DM([[1.0, 0, dt, 0], [0, 1.0, 0, dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    B = cs.DM([[0, 0], [0, 0], [dt, 0], [0, dt]])

    for k in range(N):
        # cost function
        obj += (x[k, :] - p_xf) @ Q @ (x[k, :] - p_xf).T + u[k, :] @ R @ u[k, :].T

        # state constraints
        # g = cs.horzcat(g, x[k, :])
        # g_lb = cs.horzcat(g_lb, -state_bound)
        # g_ub = cs.horzcat(g_ub, state_bound)
        #
        # # input constraints
        # g = cs.horzcat(g, u[k, :])
        # g_lb = cs.horzcat(g_lb, -input_bound)
        # g_ub = cs.horzcat(g_ub, input_bound)

        # dynamics
        g = cs.horzcat(g, (A @ x[k, :].T + B @ u[k, :].T).T - x[k + 1, :])
        g_lb = cs.horzcat(g_lb, cs.DM([0, 0, 0, 0]).T)
        g_ub = cs.horzcat(g_ub, cs.DM([0, 0, 0, 0]).T)

        # obstacle constraints
        for i in range(Nobs):
            g = cs.horzcat(g, cs.sum2((x[k, :2] - p_obs_c[i, :])**2) - p_obs_r[i, :]**2)
            g_lb = cs.horzcat(g_lb, cs.DM([0]))
            g_ub = cs.horzcat(g_ub, cs.DM.inf())

    # Terminal cost/constraints
    obj += (x[N, :] - p_xf) @ Qf @ (x[N, :] - p_xf).T

    # Initial condition
    g = cs.horzcat(g, x[0, :] - p_x0)
    g_lb = cs.horzcat(g_lb, cs.DM([0, 0, 0, 0]).T)
    g_ub = cs.horzcat(g_ub, cs.DM([0, 0, 0, 0]).T)

    # Generate solver
    x_nlp = cs.vertcat(cs.reshape(x, (N + 1) * 4, 1), cs.reshape(u, N * 2, 1))
    lbx = cs.vertcat(cs.reshape(x_lb, (N + 1) * 4, 1), cs.reshape(u_lb, N * 2, 1))
    ubx = cs.vertcat(cs.reshape(x_ub, (N + 1) * 4, 1), cs.reshape(u_ub, N * 2, 1))
    p_nlp = cs.vertcat(p_x0.T, p_xf.T, cs.reshape(p_obs_c, 2 * Nobs, 1), p_obs_r)
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

    nlp_solver = cs.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": g_lb, "ubg": g_ub, "lbx": lbx, "ubx": ubx}

    return solver


def generate_trajectory(x0, xf, N, dt, Q, R, v_max, p_max, u_max, obs, Qf=None):
    Nobs = len(obs['r'])

    nlp = trajopt_solver(N, dt, Q, R, v_max, p_max, u_max, Nobs, Qf=Qf)

    params = np.vstack([x0[:, None], xf[:, None], np.reshape(obs['c'], (2 * Nobs, 1)), obs['r'][:, None]])

    u_init = np.zeros((N, 2))
    # x_init = np.repeat(xf[:, None], N + 1, 1)
    # x_init = np.repeat(x0[:, None], N + 1, 1)
    x_init = np.outer(np.linspace(0, 1, N+1), (xf - x0)) + x0

    x_init = np.vstack([np.reshape(x_init, ((N + 1) * 4, 1)), np.reshape(u_init, (N * 2, 1))])

    tic = time.perf_counter_ns()
    sol = nlp["solver"](x0=x_init, p=params, lbg=nlp["lbg"], ubg=nlp["ubg"], lbx=nlp["lbx"], ubx=nlp["ubx"])
    toc = time.perf_counter_ns()
    print(f"Solve Time: {(toc - tic) / 1e6}ms")

    # extract solution
    x_sol = np.array(sol["x"][:(N + 1) * 4, :].reshape((N + 1, 4)))
    u_sol = np.array(sol["x"][(N + 1) * 4:, :].reshape((N, 2)))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.linspace(0, N * dt, N + 1), x_sol)
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.legend(['x', 'y', 'vx', 'vy'])

    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, (N - 1) * dt, N), u_sol)
    plt.xlabel('Time (s)')
    plt.ylabel('Input')
    plt.legend(['ux', 'uy'])
    plt.show()

    fig, ax = plt.subplots()
    for i in range(Nobs):
        circ = plt.Circle(obs['c'][:, i], obs['r'][i], color='r', alpha=0.5)
        ax.add_patch(circ)
    plt.plot(x0[0], x0[1], 'rx')
    plt.plot(xf[0], xf[1], 'go')

    plt.plot(x_sol[:, 0], x_sol[:, 1], 'k')
    plt.axis("square")
    plt.show()

    print("Complete")


if __name__ == "__main__":
    generate_trajectory(x0, xf, N, dt, Q, R, v_max, p_max, u_max, obs)
