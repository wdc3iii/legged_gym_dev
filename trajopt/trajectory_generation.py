from abc import ABC, abstractmethod

import numpy as np
import torch

class AbstractTrajectoryGenerator(ABC):
    def __init__(self, rom, device):
        self.rom = rom
        self.device = device

    def reset(self, z):
        self.reset_idx(self.arange(self.rom.n_robots), z)

    @abstractmethod
    def reset_idx(self, idx, z):
        raise NotImplementedError



class TrajectoryGenerator:

    def __init__(self, rom, t_sampler, weight_sampler, dt_loop=0.02, N=4, freq_low=0.01, freq_high=10,
                 seed=42, backend='numpy', device='cuda', prob_stationary=.01, dN=1, prob_rnd=0.05):
        self.rom = rom
        self.dt_loop = dt_loop      # Rate at which env steps
        self.device = device
        if backend == 'numpy':
            self.zeros = np.zeros
            self.ones = lambda sh, dtype: np.ones(sh, dtype=dtype)
            self.ones_like = lambda sh, dt: np.ones_like(dt, dtype=dt)
            self.zeros_like = np.zeros_like
            self.rng = np.random.default_rng(seed)
            self.arange = np.arange
            self.concatenate = lambda d, dim: np.concatenate(d, axis=dim)
            self.choice = lambda h, sh: np.random.choice(np.arrange(h), size=(*sh, 1))
            self.nonzero = lambda n: n.nonzero()[0]
            self.sin = np.sin
            self.sum = np.sum
            self.array = np.array
            self.pi = np.pi

            def uniform(low, high, size):
                return self.rng.uniform(low, high, size)
        elif backend == 'torch':
            self.zeros = lambda sh: torch.zeros(sh, device=device)
            self.zeros_type = lambda sh, ty: torch.zeros(sh, dtype=ty, device=device)
            self.ones = lambda sh, dtype: torch.ones(sh, dtype=dtype, device=device)
            self.ones_like = lambda sh, dt: torch.ones_like(dt, dtype=dt, device=device)
            self.zeros_like = lambda m: torch.zeros_like(m, device=device)
            self.arange = lambda l: torch.arange(l, device=device)
            self.concatenate = lambda d, dim: torch.concatenate(d, dim=dim)
            self.choice = lambda h, sh: torch.randint(0, h, (*sh, 1), device=device)
            self.nonzero = lambda n: torch.nonzero(n)
            self.sin = torch.sin
            self.sum = torch.sum
            self.array = lambda a: torch.tensor(a, device=device)
            self.pi = torch.pi

            def uniform(low, high, size):
                return (high - low) * torch.rand(*size, device=device) + low

        else:
            raise ValueError(f'Unsupported backend: {device}')
        self.uniform = uniform
        self.N = N
        self.dN = dN
        self.weights = self.zeros((self.rom.n_robots, 4))
        self.t_final = self.zeros((self.rom.n_robots,))
        self.t = self.zeros((self.rom.n_robots,))
        self.k = self.zeros_type((self.rom.n_robots,), torch.int)
        self.sample_hold_input = self.zeros((self.rom.n_robots, self.rom.m))
        self.extreme_input = self.zeros((self.rom.n_robots, self.rom.m))
        self.ramp_t_start = self.zeros((self.rom.n_robots,))
        self.ramp_v_start = self.zeros((self.rom.n_robots, self.rom.m))
        self.ramp_v_end = self.uniform(self.rom.v_min, self.rom.v_max, size=(self.rom.n_robots, self.rom.m))
        self.sin_mag = self.zeros((self.rom.n_robots, self.rom.m))
        self.sin_freq = self.zeros((self.rom.n_robots, self.rom.m))
        self.sin_off = self.zeros((self.rom.n_robots, self.rom.m))
        self.sin_mean = self.zeros((self.rom.n_robots, self.rom.m))
        self.t_sampler = t_sampler
        self.k_max = int(t_sampler.t_high / self.rom.dt + 2) + self.N * self.dN
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.weight_sampler = weight_sampler
        self.trajectory = self.zeros((self.rom.n_robots, self.N * self.dN + 1, self.rom.n))
        self.v_trajectory = self.zeros((self.rom.n_robots, self.N * self.dN, self.rom.m))
        self.v = self.zeros((self.rom.n_robots, self.rom.m))
        self.prob_stationary = prob_stationary
        self.stationary_inds = self.zeros((self.rom.n_robots,)).bool()
        self.prob_rnd = prob_rnd
        self.rnd_inds_bool = self.zeros((self.rom.n_robots,)).bool()
        # self.rnd_input = self.zeros((self.rom.n_robots, self.k_max, self.rom.m))

    def resample(self, idx, z):
        if len(idx) > 0:
            v_min, v_max = self.rom.compute_state_dependent_input_bounds(z[idx, :])
            self._resample_const_input(idx, v_min, v_max)
            self._resample_ramp_input(idx, z, v_min, v_max)
            self._resample_extreme_input(idx, v_min, v_max)
            self._resample_sinusoid_input(idx, v_min, v_max)
            self._resample_t_final(idx)
            self._resample_weight(idx)
            n = self.sum(idx) if idx.dtype == bool else len(idx)
            self.stationary_inds[idx] = self.uniform(self.array([0.0]), self.array([1.0]), (n, 1)).squeeze() < self.prob_stationary
            self.rnd_inds_bool[idx] = self.uniform(self.array([0.0]), self.array([1.0]),
                                                     (n, 1)).squeeze() < self.prob_rnd
            self.rnd_inds = torch.nonzero(self.rnd_inds_bool).reshape(-1,)

    def _resample_t_final(self, idx):
        self.t_final[idx] += self.t_sampler.sample(len(idx))

    def _resample_weight(self, idx):
        self.weights[idx, :] = self.weight_sampler.sample(len(idx))

    def _resample_const_input(self, idx, v_min, v_max):
        self.sample_hold_input[idx, :] = self.uniform(v_min, v_max, size=(len(idx), self.rom.m))

    def _resample_const_input(self, idx, v_min, v_max):
        self.sample_hold_input[idx, :] = self.uniform(v_min, v_max, size=(len(idx), self.rom.m))

    def _resample_ramp_input(self, idx, z, v_min, v_max):
        self.ramp_v_start[idx, :] = self.rom.clip_v_z(z[idx, :], self.ramp_v_end[idx, :])
        self.ramp_v_end[idx, :] = self.uniform(v_min, v_max, size=(len(idx), self.rom.m))
        self.ramp_t_start[idx] = self.t_final[idx]

    def _resample_extreme_input(self, t_mask,v_min, v_max):
        arr = self.concatenate((v_min[:, :, None], self.zeros_like(v_min)[:, :, None], v_max[:, :, None]), -1)
        mask = self.arange(3)[None, None, :] == self.choice(3, v_min.shape)
        self.extreme_input[t_mask, :] = arr[mask].reshape(v_min.shape)

    def _resample_sinusoid_input(self, idx, v_min, v_max):
        self.sin_mag[idx, :] = self.uniform(self.zeros_like(v_max), (v_max - v_min) / 2, size=(len(idx), self.rom.m))
        self.sin_mean[idx, :] = self.uniform(v_min + self.sin_mag[idx, :], v_max - self.sin_mag[idx, :], size=(len(idx), self.rom.m))
        self.sin_freq[idx, :] = self.uniform(self.array([self.freq_low]), self.array([self.freq_high]), size=(len(idx), self.rom.m))
        self.sin_off[idx, :] = self.uniform(self.array([-self.pi]), self.array([self.pi]), size=(len(idx), self.rom.m))

    def _const_input(self):
        return self.sample_hold_input

    def _ramp_input_t(self, t):
        return self.ramp_v_start + \
            (self.ramp_v_end - self.ramp_v_start) * ((t - self.ramp_t_start) / (self.t_final - self.ramp_t_start))[:, None]

    def _extreme_input(self):
        return self.extreme_input

    def _sinusoid_input_t(self, t):
        return self.sin_mag * self.sin(self.sin_freq * t[:, None] + self.sin_off) + self.sin_mean

    def get_input_t(self, t, z):
        idx = self.nonzero(t > self.t_final).reshape((-1,))
        self.resample(idx, z)
        return self.weights[:, 0][:, None] * self.rom.clip_v_z(z, self._const_input()) + \
            self.weights[:, 1][:, None] * self.rom.clip_v_z(z, self._ramp_input_t(t)) + \
            self.weights[:, 2][:, None] * self.rom.clip_v_z(z, self._extreme_input()) + \
            self.weights[:, 3][:, None] * self.rom.clip_v_z(z, self._sinusoid_input_t(t))

    def step(self):
        self.step_idx(self.arange(self.rom.n_robots))

    def step_idx(self, idx):
        # first, remask for envs which need a rom step
        masked_idx = idx[self.t[idx] >= self.k[idx] * self.rom.dt - 1e-5]
        self.step_rom_idx(masked_idx)
        self.t[idx] += self.dt_loop

    def step_rom_idx(self, idx, increment_rom_time=False):
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

    def reset(self, z):
        self.reset_idx(self.arange(self.rom.n_robots), z)

    def reset_idx(self, idx, z):
        self.trajectory[idx, :, :] = self.zeros((len(idx), self.N * self.dN + 1, self.rom.n))
        self.v_trajectory[idx, :, :] = self.zeros((len(idx), self.N * self.dN, self.rom.m))
        self.trajectory[idx, -1, :] = z[idx, :]
        self.k[idx] = -self.N * self.dN
        self.t[idx] = self.k[idx] * self.rom.dt
        self.t_final[idx] = self.k[idx] * self.rom.dt
        self.resample(idx, z)

        for t in range(self.N * self.dN):
            self.step_rom_idx(idx, increment_rom_time=True)

    def get_trajectory(self):
        traj0 = self.trajectory[:, :-1, :]
        traj1 = self.trajectory[:, 1:, :]
        # Interpolate time is between planning times
        traj_interp = traj0 + (traj1 - traj0) * (self.t - (self.k - 1) * self.rom.dt)[:, None, None] / self.rom.dt
        return traj_interp[:, ::self.dN, :]

    def get_v_trajectory(self):
        return self.v_trajectory[:, ::self.dN, :]
