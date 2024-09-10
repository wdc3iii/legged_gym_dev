from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
from trajopt.tube_trajopt import *
import pickle
from omegaconf import OmegaConf
from deep_tube_learning.utils import unnormalize_dict


prob_str = 'right'
# prob_str = 'right_wide'
# prob_str = 'gap'
# prob_str = 'complex'
# prob_str = 'gap_big'

track_warm = True
warm_start = 'nominal'
tube_ws = "evaluate"

# solver_str = "ipopt"
solver_str = 'snopt'

# tube_dyn = "NN_oneshot"
tube_dyn = "NN_recursive"
lim_solve = True
# # nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/jtu9xrfq"  # Hopper with ff
# nn_path = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/efctig2n"
nn_paths = [ # Hopper Recursive
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/sjiqi49f",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/7p1zump7",
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/msj97p19",
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/ks1eg0xw",
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/rg5itafm",
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/j88i9kim",
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/mnfs3r5v",
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/r5xu847t",
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/l4wnnx72",
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/w8flp57h",
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/f9zr70ds"
]


Rv1 = 10
Rv2 = 10


def main():
    for nn_path in nn_paths:
        start = problem_dict[prob_str]["start"]
        goal = problem_dict[prob_str]["goal"]
        obs = problem_dict[prob_str]["obs"]

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

        Q = problem_dict[prob_str]["Q"]
        Qw = problem_dict[prob_str]["Qw"]
        R = problem_dict[prob_str]["R"] if track_warm else problem_dict[prob_str]["R_nominal"]
        R_nominal = problem_dict[prob_str]["R_nominal"]
        Rv_first = problem_dict[prob_str]["Rv_first"] if Rv1 is None else Rv1
        Rv_second = problem_dict[prob_str]["Rv_second"] if Rv2 is None else Rv2
        N = model_cfg.dataset.H_fwd
        H_rev = model_cfg.dataset.H_rev
        w_max = 1

        tube_dynamics, _, _, eval_tube = get_tube_dynamics(tube_dyn, nn_path=nn_path)

        tube_ws_str = str(tube_ws).replace('.', '_')
        tag = "rec" if tube_dyn == "NN_recursive" else "os"
        if lim_solve:
            tag += "_lim"
        fn = f"data/{nn_path[-8:]}_{tag}.csv"
        sol, solver, t_sol, lim_sol, lim_solver, t_lim_sol = solve_tube(
            start, goal, obs, planning_model, tube_dynamics, eval_tube, N, H_rev, Q, Qw, R, w_max, R_nominal=R_nominal,
            Rv_first=Rv_first, Rv_second=Rv_second, warm_start=warm_start, tube_ws=tube_ws,
            debug_filename=fn, track_warm=track_warm, solver_str=solver_str
        )

        z_sol, v_sol, w_sol = extract_solution(sol, N, planning_model.n, planning_model.m)
        z_lim_sol, v_lim_sol, w_lim_sol = extract_solution(lim_sol, N, planning_model.n, planning_model.m)

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


        # Lim
        # Plot state space solution
        plt.subplot(3, 1, 1)
        plt.plot(z_lim_sol)
        plt.legend(["x", "y"])
        plt.xlabel("Node")
        plt.ylabel("State")

        plt.subplot(3, 1, 2)
        plt.plot(v_lim_sol)
        plt.legend(["v_x", "v_y"])
        plt.xlabel("Node")
        plt.ylabel("Input")

        plt.subplot(3, 1, 3)
        plt.plot(w_lim_sol)
        plt.legend(["w"])
        plt.xlabel("Node")
        plt.ylabel("Tube")
        plt.show()

        # Plot constraint violation
        lim_g_violation = compute_constraint_violation(lim_solver, lim_sol["g"])
        lim_g_dict = segment_constraint_violation(lim_g_violation, lim_solver["g_cols"])
        plt.figure()
        for label, data in lim_g_dict.items():
            plt.plot(data, label=label)
        plt.title("Constraint Violation")
        plt.legend()
        plt.show()

        # Plot spacial solution
        fig, ax = plt.subplots()
        plot_problem(ax, obs, start, goal)
        planning_model.plot_spacial(ax, z_lim_sol)
        planning_model.plot_tube(ax, z_lim_sol, w_lim_sol)
        plt.axis("square")
        plt.show()

        print(f"Complete! Writing to {fn}")


if __name__ == '__main__':
    main()
