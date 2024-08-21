import matplotlib.pyplot as plt
from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
from trajopt.tube_trajopt import *

# prob_str = 'right'
# prob_str = 'right_wide'
prob_str = 'gap'

vel_max = 1    # m/x
pos_max = 10   # m
dt = 0.1


def main(start, goal, obs):
    z_max = np.array([pos_max, pos_max])
    v_max = np.array([vel_max, vel_max])
    planning_model = CasadiSingleInt2D(dt, -z_max, z_max, -v_max, v_max)

    Q = 10 * np.eye(2)
    R = 0.1 * np.eye(2)
    N = 50
    solver, nlp_dict, nlp_opts = trajopt_solver(planning_model, N, Q, R, len(obs["r"]))

    v_init = np.zeros((N, planning_model.m))
    # z_init = np.repeat(zf[:, None], N + 1, 1)
    # z_init = np.repeat(z0[:, None], N + 1, 1)
    z_init = np.outer(np.linspace(0, 1, N + 1), (goal - start)) + start

    params = init_params(start, goal, obs)
    x_init = init_decision_var(z_init, v_init)

    sol = solver["solver"](x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"], lbx=solver["lbx"],
                           ubx=solver["ubx"])

    z_sol, v_sol = extract_solution(sol, N, planning_model.n, planning_model.m)

    # Plot spacial solution
    fig, ax = plt.subplots()
    plot_problem(ax, obs, start, goal)
    planning_model.plot_spacial(ax, z_sol)
    plt.axis("square")
    plt.show()

    # Plot state space solution
    plt.subplot(1, 2, 1)
    plt.plot(z_sol)
    plt.legend(["x", "y"])
    plt.xlabel("Node")
    plt.ylabel("State")

    plt.subplot(1, 2, 2)
    plt.plot(v_sol)
    plt.legend(["v_x", "v_y"])
    plt.xlabel("Node")
    plt.ylabel("Input")
    plt.show()

    # Plot constraint violation
    g = np.array(solver["solver"].get_function("nlp_g")(sol['x'], params)).T
    ubg = np.array(solver["ubg"]).T
    lbg = np.array(solver["lbg"]).T
    pass


if __name__ == '__main__':
    main(
        problem_dict[prob_str]["start"],
        problem_dict[prob_str]["goal"],
        problem_dict[prob_str]["obs"],
    )
