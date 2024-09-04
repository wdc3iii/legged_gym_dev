from trajopt.trajectory_generation import AbstractTrajectoryGenerator
from trajopt.tube_trajopt import *
from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
import torch
import time


class ClosedLoopTrajectoryGenerator(AbstractTrajectoryGenerator):

    def __init__(self, rom, H, N, dt_loop, device, p_dict, tube_dyn, nn_path=None,
                 w_max=1, mpc_dk=1, warm_start='nominal', nominal_ws="interpolate", track_nominal=True,
                 tube_ws='evaluate', max_iter=200, solver_str='ipopt'):
        super().__init__(rom, N, 1, dt_loop, device)
        self.tube_dyn_str = tube_dyn
        self.nn_path = nn_path
        self.tube_dynamics, self.H_fwd, self.H_rev, self.eval_tube = get_tube_dynamics(tube_dyn, nn_path=nn_path)
        assert self.H_fwd >= self.N
        assert self.rom.n_robots == 1
        self.H = H
        self.planning_model = CasadiSingleInt2D(
            rom.dt,
            self.rom.z_min.cpu().numpy(), self.rom.z_max.cpu().numpy(),
            self.rom.v_min.cpu().numpy(), self.rom.v_max.cpu().numpy()
        )
        self.prob_str = p_dict['name']
        self.start = p_dict['start']
        self.goal = p_dict['goal']
        self.obs = p_dict['obs']
        self.Nobs = len(self.obs['r'])
        self.Q = p_dict['Q']
        self.Qf = p_dict['Qf'] if 'Qf' in p_dict.keys() else self.Q
        self.Qw = p_dict['Qw']
        self.R = p_dict["R"] if track_nominal else p_dict["R_nominal"]
        self.R_nominal = p_dict["R_nominal"]
        self.Rv_first = p_dict["Rv_first"]
        self.Rv_second = p_dict["Rv_second"]
        self.w_max = w_max
        self.tube_ws = tube_ws
        self.warm_start = warm_start
        self.nominal_ws = nominal_ws
        self.max_iter = max_iter
        self.track_nominal = track_nominal
        self.t_wall = p_dict["t_wall"] if 't_wall' in p_dict.keys() else 10.0
        self.mpc_dk = mpc_dk

        self.z_warm, self.v_warm, self.w_warm = None, None, None
        self.nominal_z_warm, self.nominal_v_warm = None, None
        self.e = np.zeros((self.H_rev, 1))
        self.v_prev = np.zeros((self.H_rev, self.planning_model.m))
        self.g_dict = {}
        self.t_solving_nominal = 0
        self.t_solving = 0
        self.z0 = None
        self.run_data = {}

        if self.warm_start == 'nominal' or self.track_nominal:
            self.init_nominal_solver, self.nominal_nlp_dict, self.init_nominal_nlp_opts, self.nominal_solver, self.nominal_nlp_opts = trajopt_solver(
                self.planning_model, self.N, self.Q, self.R_nominal, self.Nobs, Qf=self.Qf, t_wall=p_dict['t_wall'],
                Rv_first=self.Rv_first, Rv_second=self.Rv_second, max_iter=self.max_iter, debug_filename=None
            )

        self.init_solver, self.nlp_dict, self.init_nlp_opts, self.solver, self.nlp_opts = trajopt_tube_solver(
            self.planning_model, self.tube_dynamics, self.N, self.H_rev, self.Q, self.Qw, self.R, self.w_max, self.Nobs,
            Qf=self.Qf, Rv_first=self.Rv_first, Rv_second=self.Rv_second, max_iter=self.max_iter, debug_filename="",
            t_wall=p_dict['t_wall'], solver_str=solver_str
        )

    def reset_idx(self, idx, z, e_prev=None):
        self.z0 = z.cpu().numpy().squeeze()
        self.t_solving_nominal = 0
        self.t_solving = 0
        self.k[idx] = -1
        self.t[idx] = self.k[idx] * self.rom.dt
        self.trajectory[0, 1, :] = z
        self.z_warm, self.v_warm = None, None
        self.nominal_z_warm, self.nominal_v_warm = None, None
        self.e = np.ones((self.H_rev, 1)) * np.linalg.norm(e_prev.detach().cpu().numpy())
        self.v_prev = np.zeros((self.H_rev, self.planning_model.m))
        self.step_rom_idx(idx, e_prev=e_prev, increment_rom_time=True, limit_time=False)
        self.run_data = {}


    def step_rom_idx(self, idx, e_prev=None, increment_rom_time=False, limit_time=True):
        data = {}
        # z0 = self.trajectory[0, 1, :].cpu().numpy()
        self.e[-1] = np.linalg.norm(e_prev.detach().cpu().numpy())
        print("\n\n\n\n\n\ne: ", self.e, "\n\n\n\n\n\n\n")
        data["e"] = np.copy(self.e)
        data["v_prev"] = np.copy(self.v_prev)
        if (self.k.item() + 1) % self.mpc_dk == 0:
            # Solve nominal ocp if necessary
            if self.track_nominal or (self.warm_start == 'nominal' and self.z_warm is None):
                if self.nominal_z_warm is None:
                    self.nominal_z_warm, self.nominal_v_warm = get_warm_start(self.nominal_ws, self.z0, self.goal, self.N, self.planning_model)
                nominal_params = init_params(self.z0, self.goal, self.obs)
                nominal_x_init = init_decision_var(self.nominal_z_warm, self.nominal_v_warm)
                data["nom_ws_z"] = np.copy(self.nominal_z_warm)
                data["nom_ws_v"] = np.copy(self.nominal_v_warm)
                # plt.plot(self.nominal_z_warm)
                # plt.xlabel(f'{self.k.item()}')
                # plt.show()
                if limit_time:
                    tic = time.perf_counter_ns()
                    nominal_sol = self.nominal_solver["solver"](
                        x0=nominal_x_init, p=nominal_params, lbg=self.nominal_solver["lbg"], ubg=self.nominal_solver["ubg"],
                        lbx=self.nominal_solver["lbx"], ubx=self.nominal_solver["ubx"]
                    )
                    self.t_solving_nominal += (time.perf_counter_ns() - tic) / 1e9
                else:
                    nominal_sol = self.init_nominal_solver["solver"](
                        x0=nominal_x_init, p=nominal_params, lbg=self.nominal_solver["lbg"], ubg=self.nominal_solver["ubg"],
                        lbx=self.nominal_solver["lbx"], ubx=self.nominal_solver["ubx"]
                    )
                self.nominal_z_warm, self.nominal_v_warm = extract_solution(nominal_sol, self.N, self.planning_model.n, self.planning_model.m)
                data["nom_sol_z"] = np.copy(self.nominal_z_warm)
                data["nom_sol_v"] = np.copy(self.nominal_v_warm)

                z_cost, v_cost = self.nominal_z_warm.copy(), self.nominal_v_warm.copy()
                if self.warm_start == 'nominal' and self.z_warm is None:
                    self.z_warm, self.v_warm = self.nominal_z_warm.copy(), self.nominal_v_warm.copy()
            else:
                z_cost, v_cost = None, None

            if not self.track_nominal:
                z_cost = np.repeat(self.goal[None, :], self.N + 1, axis=0)
                v_cost = np.zeros((self.N, self.rom.m))
            if self.z_warm is None:
                self.z_warm, self.v_warm = get_warm_start(
                    self.warm_start, self.z0, self.goal, self.N,self.planning_model, self.obs,
                    self.Q, self.R, nominal_ws=self.nominal_ws
                )
            params = init_params(self.z0, self.goal, self.obs, z_cost=z_cost, v_cost=v_cost, e=self.e, v_prev=self.v_prev)
            if self.w_warm is None:
                self.w_warm = get_tube_warm_start(self.tube_ws, self.eval_tube, self.z_warm, self.v_warm, self.e, self.v_prev)
            x_init = init_decision_var(self.z_warm, self.v_warm, w=self.w_warm)

            data["z_cost"] = np.copy(z_cost)
            data["v_cost"] = np.copy(v_cost)
            data["ws_z"] = np.copy(self.z_warm)
            data["ws_v"] = np.copy(self.v_warm)
            data["ws_w"] = np.copy(self.w_warm)
            if limit_time:
                tic = time.perf_counter_ns()
                sol = self.solver["solver"](
                    x0=x_init, p=params, lbg=self.solver["lbg"], ubg=self.solver["ubg"],
                    lbx=self.solver["lbx"], ubx=self.solver["ubx"]
                )
                self.t_solving += (time.perf_counter_ns() - tic) / 1e9
            else:
                sol = self.init_solver["solver"](
                    x0=x_init, p=params, lbg=self.solver["lbg"], ubg=self.solver["ubg"],
                    lbx=self.solver["lbx"], ubx=self.solver["ubx"]
                )
            self.trajectory, self.v_trajectory, self.w_trajectory = extract_solution(sol, self.N, self.planning_model.n, self.planning_model.m)
            self.z_warm, self.v_warm, self.w_warm = self.trajectory.copy(), self.v_trajectory.copy(), self.w_trajectory.copy()
            data["sol_z"] = np.copy(self.trajectory)
            data["sol_v"] = np.copy(self.v_trajectory)
            data["sol_w"] = np.copy(self.w_trajectory)

            self.trajectory = torch.from_numpy(self.trajectory[None, :, :]).float().to(self.device)
            self.v_trajectory = torch.from_numpy(self.v_trajectory[None, :, :]).float().to(self.device)

            g_violation = compute_constraint_violation(self.solver, sol["g"])
            self.g_dict = segment_constraint_violation(g_violation, self.solver["g_cols"])
            data["g_violation"] = self.g_dict.copy()

            # print debugging info
            if self.nlp_opts['iteration_callback'] is not None:
                fn = f"data/cl_tube_{self.prob_str}_{self.nn_path[-8:]}_{self.warm_start}_Rv_{self.Rv_first}_{self.Rv_second}_N_{self.N}_dk_{self.mpc_dk}_{self.tube_dyn_str}_{self.tube_ws}_{self.track_nominal}_k{self.k.item()}.csv"
                self.nlp_opts['iteration_callback'].write_data(self.solver, params, fn)
                print(f"Writing {self.k}: {fn}")
        else:
            self.trajectory[:, :-1, :] = torch.clone(self.trajectory[:, 1:, :])
            self.trajectory[:, -1, :] = self.rom.f(self.trajectory[:, -2, :], self.v_trajectory[:, -1, :])
            self.v_trajectory[:, :-1, :] = torch.clone(self.v_trajectory[:, 1:, :])
            self.w_trajectory[:-1, :] = self.w_trajectory[1:, :]
            data["sol_z"] = np.copy(self.trajectory)
            data["sol_v"] = np.copy(self.v_trajectory)
            data["sol_w"] = np.copy(self.w_trajectory)

        # Write data to file
        self.run_data[f"run{self.k.item()}"] = data
        z0_torch = torch.from_numpy(self.z0).float().to(self.device)
        self.z0 = self.rom.f(z0_torch, self.rom.clip_v_z(z0_torch, self.v_trajectory[:, 0, :])).cpu().numpy().squeeze()
        # Slide trajectories forward
        self.z_warm[:-1, :] = self.z_warm[1:, :]
        self.z_warm[-1, :] = self.rom.f(torch.from_numpy(self.z_warm[None, -2, :]).float().to(self.device), torch.from_numpy(self.v_warm[None, -1, :]).float().to(self.device)).cpu().numpy()
        self.v_warm[:-1, :] = self.v_warm[1:, :]
        self.w_warm[:-1, :] = self.w_warm[1:, :]

        # Cycle error and inputs
        self.e[:-1] = self.e[1:, :] # Note the current error is unknown, as traj_gen does not have access to it
        self.v_prev[:-1, :] = self.v_prev[1:, :]
        self.v_prev[-1, :] = self.v_trajectory[:, 0, :].cpu().numpy()

        self.k[idx] += 1
        if increment_rom_time:
            self.t[idx] += self.rom.dt

    def write_data(self, fn):
        from scipy.io import savemat
        savemat(fn, self.run_data)