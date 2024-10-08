from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
from trajopt.tube_trajopt import *

# prob_str = 'right'
# prob_str = 'right_wide'
prob_str = 'gap'

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
nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/0ks4vit2"  # 128x128 softplus b=5
# nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/idbt0oad"  # 128x128 softplus b=5 hopper



def main(start, goal, obs, vel_max, pos_max, dt):
    z_max = np.array([pos_max, pos_max])
    v_max = np.array([vel_max, vel_max])
    planning_model = CasadiSingleInt2D(dt, -z_max, z_max, -v_max, v_max)

    Q = 10 * np.eye(2)
    Qw = 0
    R = 10 * np.eye(2)
    N = 50
    H_rev = 10
    w_max = 1

    tube_dynamics = get_tube_dynamics(tube_dyn, nn_path=nn_path)

    tube_ws_str = str(tube_ws).replace('.', '_')
    fn = f"data/tube_{prob_str}_{warm_start}_{tube_dyn}_{tube_ws_str}_{track_warm}.csv"
    sol, solver = solve_tube(
        start, goal, obs, planning_model, tube_dynamics, N, H_rev, Q, Qw, R, w_max,
        warm_start=warm_start, tube_ws=tube_ws, debug_filename=fn, track_warm=track_warm
    )

    z_sol, v_sol, w_sol = extract_solution(sol, N, planning_model.n, planning_model.m)

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
    planning_model.plot_tube(ax, z_sol, w_sol)
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
