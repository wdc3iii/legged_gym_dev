from trajopt.trajectory_generation import AbstractTrajectoryGenerator
from trajopt.tube_trajopt import *
from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
import torch


class ClosedLoopTrajectoryGenerator(AbstractTrajectoryGenerator):

    def __init__(self, rom, H, dt_loop, device, p_dict, tube_dyn, nn_path=None,
                 w_max=1, warm_start='nominal', nominal_ws="interpolate", track_nominal=True, tube_ws='evaluate', max_iter=200):
        self.tube_dynamics, self.H_fwd, self.H_rev = get_tube_dynamics(tube_dyn, nn_path=nn_path)
        super().__init__(rom, self.H_fwd, 1, dt_loop, device)
        assert self.rom.n_robots == 1
        self.H = H
        self.planning_model = CasadiSingleInt2D(
            rom.dt,
            self.rom.z_min.cpu().numpy(), self.rom.z_max.cpu().numpy(),
            self.rom.v_min.cpu().numpy(), self.rom.v_max.cpu().numpy()
        )

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

        self.z_warm, self.v_warm = None, None
        self.nominal_z_warm, self.nominal_v_warm = None, None
        self.e = np.zeros((self.H_rev, 1))
        self.v_prev = np.zeros((self.H_rev, self.planning_model.m))
        self.g_dict = {}

        if self.warm_start == 'nominal' or self.track_nominal:
            self.nominal_solver, self.nominal_nlp_dict, self.nominal_nlp_opts = trajopt_solver(
                self.planning_model, self.N, self.Q, self.R_nominal, self.Nobs, Qf=self.Qf,
                Rv_first=self.Rv_first, Rv_second=self.Rv_second, max_iter=self.max_iter, debug_filename=None
            )

        self.solver, self.nlp_dict, self.nlp_opts = trajopt_tube_solver(
            self.planning_model, self.tube_dynamics, self.N, self.H_rev, self.Q, self.Qw, self.R, self.w_max, self.Nobs,
            Qf=self.Qf, Rv_first=self.Rv_first, Rv_second=self.Rv_second, max_iter=self.max_iter, debug_filename=None
        )

    def reset_idx(self, idx, z, e_prev=None):
        self.k[idx] = -1
        self.t[idx] = self.k[idx] * self.rom.dt
        self.trajectory[0, 1, :] = z
        self.z_warm, self.v_warm = None, None
        self.nominal_z_warm, self.nominal_v_warm = None, None
        self.e = np.ones((self.H_rev, 1)) * np.linalg.norm(e_prev.detach().cpu().numpy())
        self.v_prev = np.zeros((self.H_rev, self.planning_model.m))
        self.step_rom_idx(idx, e_prev=e_prev, increment_rom_time=True)

    def step_rom_idx(self, idx, e_prev=None, increment_rom_time=False):
        z0 = self.trajectory[0, 1, :].cpu().numpy()
        self.e[-1] = np.linalg.norm(e_prev.detach().cpu().numpy())
        # Solve nominal ocp if necessary
        if self.track_nominal or (self.warm_start == 'nominal' and self.z_warm is None):
            if self.nominal_z_warm is None:
                self.nominal_z_warm, self.nominal_v_warm = get_warm_start(self.nominal_ws, z0, self.goal, self.N, self.planning_model)
            nominal_params = init_params(z0, self.goal, self.obs)
            nominal_x_init = init_decision_var(self.nominal_z_warm, self.nominal_v_warm)

            nominal_sol = self.nominal_solver["solver"](
                x0=nominal_x_init, p=nominal_params, lbg=self.nominal_solver["lbg"], ubg=self.nominal_solver["ubg"],
                lbx=self.nominal_solver["lbx"], ubx=self.nominal_solver["ubx"]
            )
            self.nominal_z_warm, self.nominal_v_warm = extract_solution(nominal_sol, self.N, self.planning_model.n, self.planning_model.m)
            z_cost, v_cost = self.nominal_z_warm.copy(), self.nominal_v_warm.copy()
            if self.warm_start == 'nominal':
                self.z_warm, self.v_warm = self.nominal_z_warm.copy(), self.nominal_v_warm.copy()
        else:
            z_cost, v_cost = None, None

        if not self.track_nominal:
            z_cost = np.repeat(self.goal[None, :], self.N + 1, axis=0)
            v_cost = np.zeros((self.N, self.rom.m))
        if self.z_warm is None:
            self.z_warm, self.v_warm = get_warm_start(
                self.warm_start, z0, self.goal, self.N,self.planning_model, self.obs,
                self.Q, self.R, nominal_ws=self.nominal_ws
            )
        params = init_params(z0, self.goal, self.obs, z_cost=z_cost, v_cost=v_cost, e=self.e, v_prev=self.v_prev)
        w_warm = get_tube_warm_start(self.tube_ws, self.tube_dynamics, self.z_warm, self.v_warm, np.zeros((self.N, 1)), self.e, self.v_prev)
        x_init = init_decision_var(self.z_warm, self.v_warm, w=w_warm)
        sol = self.solver["solver"](
            x0=x_init, p=params, lbg=self.solver["lbg"], ubg=self.solver["ubg"],
            lbx=self.solver["lbx"], ubx=self.solver["ubx"]
        )
        self.trajectory, self.v_trajectory, self.w_trajectory = extract_solution(sol, self.N, self.planning_model.n, self.planning_model.m)
        self.z_warm, self.v_warm = self.trajectory.copy(), self.v_trajectory.copy()

        # Cycle error and inputs
        self.e[:-1] = self.e[1:, :] # Note the current error is unknown, as traj_gen does not have access to it
        self.v_prev[:-1, :] = self.v_prev[1:, :]
        self.v_prev[-1, :] = self.v_trajectory[0, :]

        self.trajectory = torch.from_numpy(self.trajectory[None, :, :]).float().to(self.device)
        self.v_trajectory = torch.from_numpy(self.v_trajectory[None, :, :]).float().to(self.device)

        g_violation = compute_constraint_violation(self.solver, sol["g"])
        self.g_dict = segment_constraint_violation(g_violation, self.solver["g_cols"])

        if self.nlp_opts['iteration_callback'] is not None:
            self.nlp_opts['iteration_callback'].write_data(self.solver, params)

        self.k[idx] += 1
        if increment_rom_time:
            self.t[idx] += self.rom.dt
