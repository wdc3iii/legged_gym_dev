from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
from trajopt.tube_trajopt import *
import pickle
from omegaconf import OmegaConf
from deep_tube_learning.utils import unnormalize_dict
import pandas as pd
import re

track_warm = True
warm_start = 'nominal'
tube_ws = "evaluate"

# solver_str = "ipopt"
solver_str = 'snopt'

# tube_dyn = "NN_oneshot"
tube_dyn = "NN_recursive"
lim_solve = True

nn_paths = [ # Hopper Recursive
    # "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/sjiqi49f",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/7p1zump7",
    "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/msj97p19",
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

N_sols = 100

Q = 10
Qw = 0
Qf = Q
R = 1
R_nominal = 0.1
Rv_first = 10
Rv_second = 10
N = 25
w_max = 1
Nobs = 3
r_min = 0.1
r_max = 0.375

planning_model = CasadiSingleInt2D(
    0.1,
    np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]),
    np.array([-0.2, -0.2]), np.array([0.2, 0.2]),
)
z0 = np.array([0., 0.])
goal = np.array([1., 0.])

def extract_snopt_table(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

     # Adjusted header pattern to handle varying whitespace and leading spaces
    header_pattern = r'^\s*Itns\s+Major\s+Minors\s+Step\s+nCon\s+Feasible\s+Optimal\s+MeritFunction\s+L\+U\s+BSwap\s+nS\s+condZHZ\s+Penalty'
    # Adjusted pattern to capture table rows with flexible spacing
    table_pattern = r'^\s*(\d+)\s+(\d+)\s+(\d+)\s*([\d.E+-]*)\s+(\d+)\s*([\d.E+-]+)\s*([\d.E+-]+)\s*([\d.E+-]+)\s+(\d+)\s*([\d.E+-]*)\s*(\d+)\s*([\d.E+-]+)\s*([\d.E+-]*)\s*(\S*)'

    # Initialize variables
    tables = []
    current_table = []
    header_found = False

    for line in data:
        # Check if the line matches the table header
        if re.match(header_pattern, line):
            # If a header is found and there is a current table, save it
            if current_table:
                tables.append(current_table)
                current_table = []  # Reset for the next table
            header_found = True

        # If header is found, collect lines that match the table pattern
        elif header_found and re.match(table_pattern, line):
            match = re.findall(table_pattern, line)
            current_table.extend(match)

    # Append the last table if exists
    if current_table:
        tables.append(current_table)

    # Define columns for the DataFrame
    columns = ['Itns', 'Major', 'Minors', 'Step', 'nCon', 'Feasible', 'Optimal',
               'MeritFunction', 'L+U', 'BSwap', 'nS', 'condZHZ', 'Penalty', 'Status']

    # If tables list is empty, handle it gracefully
    if not tables:
        print("No tables found in the file.")
        return pd.DataFrame(columns=columns)

    # Combine all tables into a single DataFrame
    all_dataframes = [pd.DataFrame(table, columns=columns) for table in tables]
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Convert numeric columns to appropriate types
    combined_df = combined_df.apply(pd.to_numeric, errors='ignore')

    return combined_df


def solve_prob(solver, nominal_solver, eval_tube, e, v_prev, obs):
    nominal_z_warm, nominal_v_warm = get_warm_start('interpolate', z0, goal, N,
                                                                  planning_model)
    nominal_params = init_params(z0, goal, obs)
    nominal_x_init = init_decision_var(nominal_z_warm, nominal_v_warm)
    
    nominal_sol = nominal_solver["solver"](
            x0=nominal_x_init, p=nominal_params, lbg=nominal_solver["lbg"], ubg=nominal_solver["ubg"],
            lbx=nominal_solver["lbx"], ubx=nominal_solver["ubx"]
        )
    nominal_z_sol, nominal_v_sol = extract_solution(nominal_sol, N, planning_model.n,
                                                                planning_model.m)

    params = init_params(z0, goal, obs, z_cost=nominal_z_sol, v_cost=nominal_v_sol, e=e, v_prev=v_prev)
    w_warm = get_tube_warm_start(tube_ws, eval_tube, nominal_z_sol, nominal_v_sol, e, v_prev)
    x_init = init_decision_var(nominal_z_sol, nominal_v_sol, w=w_warm)

    tic = time.perf_counter_ns()
    sol = solver["solver"](
        x0=x_init, p=params, lbg=solver["lbg"], ubg=solver["ubg"],
        lbx=solver["lbx"], ubx=solver["ubx"]
    )
    t_solving = (time.perf_counter_ns() - tic) / 1e9

    z_sol, v_sol, w_sol = extract_solution(sol, N, planning_model.n, planning_model.m)
    fig, ax = plt.subplots()
    plot_problem(ax, obs, z0, goal)
    planning_model.plot_spacial(ax, z_sol)
    planning_model.plot_tube(ax, z_sol, w_sol)
    plt.axis("square")
    plt.show()

    iter_info = extract_snopt_table("trajectory_generator.out")
    return {
        "solve time": t_solving,
        "z_sol": z_sol,
        "v_sol": v_sol,
        "w_sol": w_sol,
        "obs": obs,
        "z0": z0,
        "goal": goal,
        "iter_info": iter_info
    }


def main():
    np.random.seed(0)
    angs = np.random.uniform(-np.pi / 2, np.pi / 2, size=(Nobs + 1, N_sols))
    dists = np.random.uniform(r_max + 0.01, 1 - r_max, size=(Nobs, N_sols))
    rs = np.random.uniform(r_min, r_max, (Nobs, N_sols))
    solver_data = []
    
    for nn_path in nn_paths:
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

        H_rev = model_cfg.dataset.H_rev

        tube_dynamics, _, _, eval_tube = get_tube_dynamics(tube_dyn, nn_path=nn_path)

        
        nom_solver, nominal_nlp_dict, nominal_nlp_opts, _, _ = trajopt_solver(
            planning_model, N, Q, R_nominal, Nobs, Qf=Qf, t_wall=0.1,
            Rv_first=Rv_first, Rv_second=Rv_second, max_iter=1000, debug_filename=None
        )

        solver, nlp_dict, nlp_opts, _, _ = trajopt_tube_solver(
            planning_model, tube_dynamics, N, H_rev, Q, Qw, R, w_max, Nobs,
            Qf=Qf, Rv_first=Rv_first, Rv_second=Rv_second, max_iter=1000, debug_filename="",
            t_wall=0.1, solver_str=solver_str, lim_sol=False
        )

        solver_data.append((solver, nom_solver, eval_tube, H_rev))

        for i in range(N_sols):
            # sample a random problem
            ang = angs[:, i]
            dist = dists[:, i]

            obs = {
                'cx': np.cos(ang[:-1]) * dist,
                'cy': np.sin(ang[:-1]) * dist,
                'r': rs[:, i]
            }

            # goal = np.array([np.cos(ang[-1]), np.sin(ang[-1])]) * 2

            for solver, nom_solver, eval_tube, H_r in solver_data:
                e_prev = np.zeros((H_r, 1))
                v_prev = np.zeros((H_r, 2))
                d = solve_prob(solver, nom_solver, eval_tube, e_prev, v_prev, obs)



if __name__ == '__main__':
    main()
