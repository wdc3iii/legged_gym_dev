from abc import ABC, abstractmethod
import torch


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

    def reset(self, z):
        self.reset_idx(torch.arange(self.rom.n_robots, device=self.device), z)

    @abstractmethod
    def reset_idx(self, idx, z):
        raise NotImplementedError

    def step(self, dt_tol=1e-5):
        self.step_idx(torch.arange(self.rom.n_robots, device=self.device), dt_tol=dt_tol)

    def step_idx(self, idx, dt_tol=1e-5):
        # first, remask for envs which need a rom step
        masked_idx = idx[self.t[idx] >= self.k[idx] * self.rom.dt - dt_tol]
        self.step_rom_idx(masked_idx)
        self.t[idx] += self.dt_loop

    @abstractmethod
    def step_rom_idx(self, idx):
        raise NotImplementedError

    def get_trajectory(self):
        traj0 = self.trajectory[:, :-1, :]
        traj1 = self.trajectory[:, 1:, :]
        # Interpolate time is between planning times
        traj_interp = traj0 + (traj1 - traj0) * (self.t - (self.k - 1) * self.rom.dt)[:, None, None] / self.rom.dt
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

    def reset_idx(self, idx, z):
        self.trajectory[idx, :, :] = torch.zeros((len(idx), self.N * self.dN + 1, self.rom.n), device=self.device)
        self.v_trajectory[idx, :, :] = torch.zeros((len(idx), self.N * self.dN, self.rom.m), device=self.device)
        self.trajectory[idx, -1, :] = z[idx, :]
        self.k[idx] = -self.N * self.dN
        self.t[idx] = self.k[idx] * self.rom.dt
        self.t_final[idx] = self.k[idx] * self.rom.dt
        self.resample(idx, z)

        for t in range(self.N * self.dN):
            self.step_rom_idx(idx, increment_rom_time=True)
