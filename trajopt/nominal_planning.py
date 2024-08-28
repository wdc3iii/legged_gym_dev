from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
from trajopt.tube_trajopt import *

# prob_str = 'right'
# prob_str = 'right_wide'
prob_str = 'gap'

# warm_start = 'start'
warm_start = 'goal'
# warm_start = 'interpolate'


def main(start, goal, obs, vel_max, pos_max, dt):
    z_max = np.array([pos_max, pos_max])
    v_max = np.array([vel_max, vel_max])
    planning_model = CasadiSingleInt2D(dt, -z_max, z_max, -v_max, v_max)

    Q = 10 * np.eye(2)
    R = 0.1 * np.eye(2)
    N = 50

    fn = f"data/nominal_{prob_str}_{warm_start}.csv"
    sol, solver = solve_nominal(start, goal, obs, planning_model, N, Q, R, warm_start=warm_start, debug_filename=fn)

    z_sol, v_sol = extract_solution(sol, N, planning_model.n, planning_model.m)

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
    g_violation = compute_constraint_violation(solver, sol["g"])
    g_dict = segment_constraint_violation(g_violation, solver["g_cols"])
    plt.figure()
    for label, data in g_dict.items():
        plt.plot(data, label=label)
    plt.title("Constraint Violation")
    plt.legend()
    plt.show()

    # Plot spacial solution
    fig, ax = plt.subplots()
    plot_problem(ax, obs, start, goal)
    planning_model.plot_spacial(ax, z_sol)
    plt.axis("square")
    plt.show()

    print("Complete!")


if __name__ == '__main__':
    main(
        problem_dict[prob_str]["start"],
        problem_dict[prob_str]["goal"],
        problem_dict[prob_str]["obs"],
        problem_dict[prob_str]["vel_max"],
        problem_dict[prob_str]["pos_max"],
        problem_dict[prob_str]["dt"]
    )