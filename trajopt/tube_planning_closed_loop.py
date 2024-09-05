from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
from trajopt.tube_trajopt import *
from trajopt.rom_dynamics import SingleInt2D
from trajopt.rom_dynamics import DoubleInt2D
import time

# prob_str = 'right'
# prob_str = 'right_wide'
prob_str = 'gap'

# warm_start = 'start'
# warm_start = 'goal'
# warm_start = 'interpolate'
warm_start = 'nominal'

# tube_ws = 0
tube_ws = "evaluate"

# tube_dyn = 'l1'
# tube_dyn = "l2"
# tube_dyn = "l1_rolling"
# tube_dyn = "l2_rolling"
tube_dyn = "NN_oneshot"
# nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/t3b8qehd"
nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/vv703y5s"

H = 75

Kp = 10
Kd = 10

time_it = True

v_max = 0.2
p_max = 1e9
dt = 0.1


def main(start, goal, obs, vel_max, pos_max, dt):
    z_max = np.array([pos_max, pos_max])
    v_max = np.array([vel_max, vel_max])
    planning_model = CasadiSingleInt2D(dt, -z_max, z_max, -v_max, v_max)
    np_pm = SingleInt2D(dt, -z_max, z_max, -v_max, v_max, backend='numpy')

    Q = 10 * np.eye(2)
    Qw = 0
    R = 10 * np.eye(2)
    N = 50
    H_rev = 25
    w_max = 1

    double_z_max = np.array([np.inf, np.inf, 0.3, 0.3])
    double_v_max = np.array([0.5, 0.5])
    double_int = DoubleInt2D(dt, -double_z_max, double_z_max, -double_v_max, double_v_max, n_robots=1, backend='numpy')

    tube_dynamics = get_tube_dynamics(tube_dyn, nn_path=nn_path)

    tube_ws_str = str(tube_ws).replace('.', '_')

    z_k = np.zeros((H + 1, planning_model.n)) * np.nan
    v_k = np.zeros((H, planning_model.m)) * np.nan
    e = np.zeros((H_rev, 1))
    v_prev = np.zeros((H_rev, planning_model.m))
    x = np.zeros((1, z_k.shape[0], double_int.n)) * np.nan
    u = np.zeros((1, v_k.shape[0], double_int.m)) * np.nan
    w_k = np.zeros((H + 1, 1)) * np.nan
    w_k[0] = 0
    pz_x = np.zeros_like(z_k) * np.nan
    z_k[0, :] = start
    x[:, 0, :2] = start
    x[:, 0, 2:] = 0
    pz_x[0, :] = double_int.proj_z(x[:, 0, :])

    # mats for visualizing later
    z_vis = np.zeros((H, *z_k.shape))
    v_vis = np.zeros((H, *v_k.shape))
    pz_x_vis = np.zeros((H, *pz_x.shape))
    w_vis = np.zeros((H, *w_k.shape))
    z_sol_vis = np.zeros((H, N + 1, planning_model.n))
    v_sol_vis = np.zeros((H, N, planning_model.m))
    w_sol_vis = np.zeros((H, N + 1, 1))
    cv_vis = {}
    timing = np.zeros((H,))
    t0 = time.perf_counter_ns()

    sol, solver = solve_tube(
        start, goal, obs, planning_model, tube_dynamics, N, H_rev, Q, Qw, R, w_max,
        warm_start=warm_start, tube_ws=tube_ws, max_iter=200
    )

    for k in range(H):
        z_sol, v_sol, w_sol = extract_solution(sol, N, planning_model.n, planning_model.m)

        # Decide fom action
        xt = x[:, k, :]
        zt = z_sol[0, :]
        vt_p = v_sol[1, :]
        ut = double_int.clip_v_z(xt, Kp * (zt - xt[:, :2]) + Kd * (vt_p - xt[:, 2:]))

        xt_p1 = double_int.f(xt, ut)

        v_k[k, :] = v_sol[0, :]
        z_k[k + 1, :] = np_pm.f(z_k[k, :], v_k[k, :])
        x[:, k + 1, :] = xt_p1
        u[:, k, :] = ut
        pz_x[k + 1, :] = double_int.proj_z(xt_p1)
        w_k[k + 1, :] = w_sol[1, :]

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
            planning_model.plot_tube(ax, z_k, w_k, 'k')
            planning_model.plot_spacial(ax, z_k, '.-k')
            planning_model.plot_spacial(ax, pz_x, '.-b')
            planning_model.plot_spacial(ax, z_sol)
            planning_model.plot_tube(ax, z_sol, w_sol)
            plt.axis("square")
            plt.show()

        z_vis[k] = z_k.copy()
        v_vis[k] = v_k.copy()
        pz_x_vis[k] = pz_x.copy()
        w_vis[k] = w_k.copy()
        z_sol_vis[k] = z_sol.copy()
        v_sol_vis[k] = v_sol.copy()
        w_sol_vis[k] = w_sol.copy()
        cv_vis["cv" + str(k)] = g_dict.copy()
        timing[k] = time.perf_counter_ns() - t0

        params = init_params(z_k[k + 1, :], goal, obs)
        e[:-1] = e[1, :]
        e[-1] = np.linalg.norm(z_k[k, :] - pz_x[k, :])
        v_prev[:-1, :] = v_prev[1:, :]
        v_prev[-1, :] = v_k[k, :]
        params = np.vstack([params, e, v_prev.reshape(-1, 1)])
        x_init = init_decision_var(z_sol, v_sol, w=w_sol)

        sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"],
                               ubx=solver["ubx"])

    from scipy.io import savemat
    fn = f"data/cl_tube_{prob_str}_{warm_start}_{tube_dyn}_{tube_ws_str}.mat"
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
        "z0": start,
        "zf": goal,
        "obs_x": obs['c'][0, :],
        "obs_y": obs['c'][1, :],
        "obs_r": obs['r'],
        "timing": timing
    })
    print("Complete!")


if __name__ == '__main__':
    main(
        problem_dict[prob_str]["start"],
        problem_dict[prob_str]["goal"],
        problem_dict[prob_str]["obs"],
        v_max,
        p_max,
        dt
    )
