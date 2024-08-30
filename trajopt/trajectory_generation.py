from abc import ABC, abstractmethod
import torch
import numpy as np
import time
from trajopt.rom_dynamics import SingleInt2D, DoubleInt2D
from trajopt.casadi_rom_dynamics import CasadiSingleInt2D
from trajopt.tube_trajopt2 import *


class AbstractTrajectoryGenerator(ABC):
    def __init__(self, rom, N, dN, dt_loop, device):
        self.rom = rom
        self.device = device
        self.t = torch.zeros((self.rom.n_robots,), device=self.device)
        self.k = torch.zeros((self.rom.n_robots,), dtype=torch.int, device=self.device)
        self.N = N
        self.dN = dN
        self.dt_loop = dt_loop
        self.trajectory = torch.zeros((self.rom.n_robots, self.N * self.dN + 1, self.rom.n), device=self.device)
        self.v_trajectory = torch.zeros((self.rom.n_robots, self.N * self.dN, self.rom.m), device=self.device)

    def uniform(self, low, high, size):
        return (high - low) * torch.rand(*size, device=self.device) + low

    def reset(self, z, e_prev=None):
        self.reset_idx(torch.arange(self.rom.n_robots, device=self.device), z, e_prev=e_prev)

    @abstractmethod
    def reset_idx(self, idx, z, e_prev=None):
        raise NotImplementedError

    def step(self, e_prev=None, dt_tol=1e-5):
        self.step_idx(torch.arange(self.rom.n_robots, device=self.device), e_prev=e_prev, dt_tol=dt_tol)

    def step_idx(self, idx, e_prev=None, dt_tol=1e-5):
        # first, remask for envs which need a rom step
        # TODO: check how changing this to k from k+1 impacts TrajectoryGenerator
        masked_idx = idx[self.t[idx] >= (self.k[idx] + 1) * self.rom.dt - dt_tol]
        if len(masked_idx):
            self.step_rom_idx(masked_idx, e_prev=e_prev)
        self.t[idx] += self.dt_loop

    @abstractmethod
    def step_rom_idx(self, idx, e_prev=None, increment_rom_time=False):
        raise NotImplementedError

    def get_trajectory(self):
        traj0 = self.trajectory[:, :-1, :]
        traj1 = self.trajectory[:, 1:, :]
        # Interpolate time is between planning times
        # TODO: check how changing this to k from k-1 impacts TrajectoryGenerator
        traj_interp = traj0 + (traj1 - traj0) * (self.t - self.k * self.rom.dt)[:, None, None] / self.rom.dt
        return traj_interp[:, ::self.dN, :]

    def get_v_trajectory(self):
        return self.v_trajectory[:, ::self.dN, :]


class TrajectoryGenerator(AbstractTrajectoryGenerator):

    def __init__(self, rom, t_sampler, weight_sampler, dt_loop=0.02, N=4, freq_low=0.01, freq_high=10,
                 device='cuda', prob_stationary=.01, dN=1, prob_rnd=0.05):
        super().__init__(rom, N, dN, dt_loop, device)

        self.weights = torch.zeros((self.rom.n_robots, 4), device=self.device)
        self.t_final = torch.zeros((self.rom.n_robots,), device=self.device)
        self.sample_hold_input = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.extreme_input = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.ramp_t_start = torch.zeros((self.rom.n_robots,), device=self.device)
        self.ramp_v_start = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.ramp_v_end = self.uniform(self.rom.v_min, self.rom.v_max, size=(self.rom.n_robots, self.rom.m))
        self.sin_mag = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.sin_freq = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.sin_off = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.sin_mean = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.t_sampler = t_sampler
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.weight_sampler = weight_sampler
        self.v = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.prob_stationary = prob_stationary
        self.stationary_inds = torch.zeros((self.rom.n_robots,), device=self.device).bool()
        self.prob_rnd = prob_rnd
        self.rnd_inds_bool = torch.zeros((self.rom.n_robots,), device=self.device).bool()

    def resample(self, idx, z):
        if len(idx) > 0:
            v_min, v_max = self.rom.compute_state_dependent_input_bounds(z[idx, :])
            self._resample_const_input(idx, v_min, v_max)
            self._resample_ramp_input(idx, z, v_min, v_max)
            self._resample_extreme_input(idx, v_min, v_max)
            self._resample_sinusoid_input(idx, v_min, v_max)
            self._resample_t_final(idx)
            self._resample_weight(idx)
            n = torch.sum(idx) if idx.dtype == bool else len(idx)
            self.stationary_inds[idx] = self.uniform(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device), (n, 1)).squeeze() < self.prob_stationary
            self.rnd_inds_bool[idx] = self.uniform(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device),
                                                     (n, 1)).squeeze() < self.prob_rnd
            self.rnd_inds = torch.nonzero(self.rnd_inds_bool).reshape(-1,)

    def _resample_t_final(self, idx):
        self.t_final[idx] += self.t_sampler.sample(len(idx))

    def _resample_weight(self, idx):
        self.weights[idx, :] = self.weight_sampler.sample(len(idx))

    def _resample_const_input(self, idx, v_min, v_max):
        self.sample_hold_input[idx, :] = self.uniform(v_min, v_max, size=(len(idx), self.rom.m))

    def _resample_ramp_input(self, idx, z, v_min, v_max):
        self.ramp_v_start[idx, :] = self.rom.clip_v_z(z[idx, :], self.ramp_v_end[idx, :])
        self.ramp_v_end[idx, :] = self.uniform(v_min, v_max, size=(len(idx), self.rom.m))
        self.ramp_t_start[idx] = self.t_final[idx]

    def _resample_extreme_input(self, t_mask,v_min, v_max):
        arr = torch.concatenate((v_min[:, :, None], torch.zeros_like(v_min, device=self.device)[:, :, None], v_max[:, :, None]), dim=-1)
        mask = torch.arange(3, device=self.device)[None, None, :] == torch.randint(0, 3, (*v_min.shape, 1), device=self.device)
        self.extreme_input[t_mask, :] = arr[mask].reshape(v_min.shape)

    def _resample_sinusoid_input(self, idx, v_min, v_max):
        self.sin_mag[idx, :] = self.uniform(torch.zeros_like(v_max), (v_max - v_min) / 2, size=(len(idx), self.rom.m))
        self.sin_mean[idx, :] = self.uniform(v_min + self.sin_mag[idx, :], v_max - self.sin_mag[idx, :], size=(len(idx), self.rom.m))
        self.sin_freq[idx, :] = self.uniform(
            torch.tensor([self.freq_low], device=self.device),
            torch.tensor([self.freq_high], device=self.device),
            size=(len(idx), self.rom.m))
        self.sin_off[idx, :] = self.uniform(
            torch.tensor([-torch.pi], device=self.device),
            torch.tensor([torch.pi], device=self.device),
            size=(len(idx), self.rom.m)
        )

    def _const_input(self):
        return self.sample_hold_input

    def _ramp_input_t(self, t):
        return self.ramp_v_start + \
            (self.ramp_v_end - self.ramp_v_start) * ((t - self.ramp_t_start) / (self.t_final - self.ramp_t_start))[:, None]

    def _extreme_input(self):
        return self.extreme_input

    def _sinusoid_input_t(self, t):
        return self.sin_mag * torch.sin(self.sin_freq * t[:, None] + self.sin_off) + self.sin_mean

    def get_input_t(self, t, z):
        idx = torch.nonzero(t > self.t_final).reshape((-1,))
        self.resample(idx, z)
        return self.weights[:, 0][:, None] * self.rom.clip_v_z(z, self._const_input()) + \
            self.weights[:, 1][:, None] * self.rom.clip_v_z(z, self._ramp_input_t(t)) + \
            self.weights[:, 2][:, None] * self.rom.clip_v_z(z, self._extreme_input()) + \
            self.weights[:, 3][:, None] * self.rom.clip_v_z(z, self._sinusoid_input_t(t))


    def step_rom_idx(self, idx, e_prev=None, increment_rom_time=False):
        # Get input to apply for trajectory
        v = self.get_input_t(self.t, self.trajectory[:, -1, :])
        v[self.stationary_inds, :] = 0
        v_min, v_max = self.rom.compute_state_dependent_input_bounds(self.trajectory[self.rnd_inds, -1, :])
        v[self.rnd_inds, :] = self.uniform(v_min, v_max, size=(len(self.rnd_inds), self.rom.m))
        self.v[idx, :] = v[idx, :]
        z_next = self.rom.f(self.trajectory[idx, -1, :], self.v[idx, :])
        mask = self.stationary_inds[:, None] & self.rom.vel_inds
        z_next[mask[idx, :]] = 0
        self.trajectory[idx, :-1, :] = self.trajectory[idx, 1:, :]
        self.trajectory[idx, -1, :] = z_next
        self.v_trajectory[idx, :-1, :] = self.v_trajectory[idx, 1:, :]
        self.v_trajectory[idx, -1, :] = self.v[idx]
        self.k[idx] += 1
        if increment_rom_time:
            self.t[idx] += self.rom.dt

    def reset_idx(self, idx, z, e_prev=None):
        self.trajectory[idx, :, :] = torch.zeros((len(idx), self.N * self.dN + 1, self.rom.n), device=self.device)
        self.v_trajectory[idx, :, :] = torch.zeros((len(idx), self.N * self.dN, self.rom.m), device=self.device)
        self.trajectory[idx, -1, :] = z[idx, :]
        self.k[idx] = -self.N * self.dN
        self.t[idx] = self.k[idx] * self.rom.dt
        self.t_final[idx] = self.k[idx] * self.rom.dt
        self.resample(idx, z)

        for t in range(self.N * self.dN):
            self.step_rom_idx(idx, increment_rom_time=True)


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
        self.R = p_dict['R']
        self.w_max = w_max
        self.tube_ws = tube_ws
        self.warm_start = warm_start
        self.nominal_ws = nominal_ws
        self.max_iter = max_iter
        self.track_nominal = track_nominal

        self.z_warm, self.v_warm = None, None
        self.e = np.zeros((self.H_rev, 1))
        self.v_prev = np.zeros((self.H_rev, self.planning_model.m))
        self.g_dict = {}

        if self.warm_start == 'nominal' or self.track_nominal:
            self.nominal_solver, self.nominal_nlp_dict, self.nominal_nlp_opts = trajopt_solver(
                self.planning_model, self.N, self.Q, self.R, self.Nobs, Qf=self.Qf,
                max_iter=self.max_iter, debug_filename=None
            )

        self.solver, self.nlp_dict, self.nlp_opts = trajopt_tube_solver(
            self.planning_model, self.tube_dynamics, self.N, self.H_rev, self.Q, self.Qw, self.R, self.w_max, self.Nobs,
            Qf=self.Qf, max_iter=self.max_iter, debug_filename=None, z_init=None, v_init=None
        )

    def reset_idx(self, idx, z, e_prev=None):
        self.k[idx] = -1
        self.t[idx] = self.k[idx] * self.rom.dt
        self.trajectory[0, 1, :] = z
        self.z_warm, self.v_warm = None, None
        self.e = np.ones((self.H_rev, 1)) * np.linalg.norm(e_prev.detach().cpu().numpy())
        self.v_prev = np.zeros((self.H_rev, self.planning_model.m))
        self.step_rom_idx(idx, e_prev=e_prev, increment_rom_time=True)

    def step_rom_idx(self, idx, e_prev=None, increment_rom_time=False):
        z0 = self.trajectory[0, 1, :].cpu().numpy()
        self.e[-1] = np.linalg.norm(e_prev.detach().cpu().numpy())
        # Solve nominal ocp if necessary
        if self.track_nominal or (self.warm_start == 'nominal' and self.z_warm is None):
            z_nominal_init, v_nominal_init = get_warm_start(self.nominal_ws, z0, self.goal, self.N, self.planning_model)
            nominal_params = init_params(z0, self.goal, self.obs)
            nominal_x_init = init_decision_var(z_nominal_init, v_nominal_init)

            nominal_sol = self.nominal_solver["solver"](
                x0=nominal_x_init, p=nominal_params, lbg=self.nominal_solver["lbg"], ubg=self.nominal_solver["ubg"],
                lbx=self.nominal_solver["lbx"], ubx=self.nominal_solver["ubx"]
            )
            z_cost, v_cost = extract_solution(nominal_sol, self.N, self.planning_model.n, self.planning_model.m)
            if self.warm_start == 'nominal':
                self.z_warm, self.v_warm = z_cost.copy(), v_cost.copy()
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


class CircleTrajectoryGenerator(TrajectoryGenerator):

    def resample(self, idx, z):
        self.center = torch.clone(z.detach())[:, :2]
        self.center[:, 0] -= 0.5

    def get_input_t(self, t, z):
        v = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        if isinstance(self.rom, SingleInt2D):
            e = z - self.center
            v[:, 0] = - e[:, 1]
            v[:, 1] = e[:, 0]
            v += -(e - 0.5 * e / torch.linalg.norm(v, dim=-1, keepdim=True))
            v = v / torch.linalg.norm(v, dim=-1, keepdim=True) * torch.min(torch.minimum(self.rom.v_max, torch.abs(self.rom.v_min)))
        elif isinstance(self.rom, DoubleInt2D):
            m = torch.min(torch.minimum(self.rom.v_max, torch.abs(self.rom.v_min)))
            z_des = self.center + 0.5 * torch.concatenate([torch.cos(t / m)[:, None], torch.sin(t / m)[:, None]], dim=-1)
            v_des = 0.5 * torch.concatenate([-torch.sin(t / m)[:, None], torch.cos(t / m)[:, None]], dim=-1) / m
            v = self.rom.clip_v_z(z, -4 * (z[:, :2] - z_des) - 4 * (z[:, 2:] - v_des))
        else:
            raise ValueError("Only SingleInt2D and DoubleInt2D are supported")
        return v


class ZeroTrajectoryGenerator(TrajectoryGenerator):

    def resample(self, idx, z):
        self.stationary_inds[idx] = True

    def get_input_t(self, t, z):
        return torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)


class SquareTrajectoryGenerator(TrajectoryGenerator):

    def resample(self, idx, z):
        pass

    def get_input_t(self, t, z):
        v = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        if isinstance(self.rom, SingleInt2D):
            c1 = 2 / self.rom.v_max[1]
            c2 = c1 + 1 / self.rom.v_max[0]
            c3 = c2 + 2 / abs(self.rom.v_min[1])
            c4 = c3 + 1 / abs(self.rom.v_min[0])
            v[(0 <= t) & (t < c1), 1] = self.rom.v_max[1] / 2
            v[(c1 <= t) & (t < c2), 0] = self.rom.v_max[0]
            v[(c2 <= t) & (t < c3), 1] = self.rom.v_min[1] / 2
            v[(c3 <= t) & (t < c4), 0] = self.rom.v_min[1]
        elif isinstance(self.rom, DoubleInt2D):
            c0 = self.rom.z_max[3] / 2 / self.rom.v_max[1]
            c1 = c0 + (1 - 2 * (0.5 * self.rom.v_max[1] * c0**2)) / (self.rom.z_max[3] / 2)
            c2 = c1 + self.rom.z_min[3] / 2 / self.rom.v_min[1]
            c3 = c2
            c4 = c3 + self.rom.z_max[2] / self.rom.v_max[0]
            c5 = c4 + (1 - 2 * (0.5 * self.rom.v_max[0] * (c4 - c3) ** 2)) / (self.rom.z_max[2] / 2)
            c6 = c5 + self.rom.z_min[2] / self.rom.v_min[0]
            c7 = c6
            c8 = c7 + self.rom.z_min[3] / 2 / self.rom.v_min[1]
            c9 = c8 + (1 - 2 * (0.5 * abs(self.rom.v_min[1]) * (c8 - c7) ** 2)) / (abs(self.rom.z_min[3]) / 2)
            c10 = c9 + self.rom.z_max[3] / 2 / self.rom.v_max[1]
            c11 = c10
            c12 = c11 + self.rom.z_min[2] / self.rom.v_min[0]
            c13 = c12 + (1 - 2 * (0.5 * abs(self.rom.v_min[0]) * (c12 - c11) ** 2)) / (abs(self.rom.z_min[2]) / 2)
            c14 = c13 + self.rom.z_max[2] / self.rom.v_max[0]

            v[(0 <= t) & (t < c0), 1] = self.rom.v_max[1].float()
            v[(c1 <= t) & (t < c2), 1] = self.rom.v_min[1].float()
            v[(c3 <= t) & (t < c4), 0] = self.rom.v_max[0].float()
            v[(c5 <= t) & (t < c6), 0] = self.rom.v_min[0].float()
            v[(c7 <= t) & (t < c8), 1] = self.rom.v_min[1].float()
            v[(c9 <= t) & (t < c10), 1] = self.rom.v_max[1].float()
            v[(c11 <= t) & (t < c12), 0] = self.rom.v_min[0].float()
            v[(c13 <= t) & (t < c14), 0] = self.rom.v_max[0].float()
        else:
            raise ValueError("Only SingleInt2D and DoubleInt2D are supported")
        return v

    def reset_idx(self, idx, z):
        z[:, self.rom.vel_inds] = 0
        super().reset_idx(idx, z)
