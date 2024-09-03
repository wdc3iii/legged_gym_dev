from abc import ABC, abstractmethod
import torch
from trajopt.rom_dynamics import SingleInt2D, DoubleInt2D


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
        traj_interp = traj0 + (traj1 - traj0) * (self.t - self.k * self.rom.dt)[:, None, None] / self.rom.dt
        return traj_interp[:, ::self.dN, :]

    def get_v_trajectory(self):
        return self.v_trajectory[:, ::self.dN, :]


class TrajectoryGenerator(AbstractTrajectoryGenerator):

    def __init__(self, rom, t_sampler, weight_sampler, dt_loop=0.02, N=4, freq_low=0.01, freq_high=10,
                 device='cuda', prob_stationary=.01, dN=1, prob_rnd=0.05, noise_max_std=0.1, noise_llh=0.25):
        super().__init__(rom, N, dN, dt_loop, device)

        self.weights = torch.zeros((self.rom.n_robots, self.rom.m, 4), device=self.device)
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
        self.rnd_mag = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.rnd_mean = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.noise_std = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.t_sampler = t_sampler
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.weight_sampler = weight_sampler
        self.v = torch.zeros((self.rom.n_robots, self.rom.m), device=self.device)
        self.prob_stationary = prob_stationary
        self.stationary_inds = torch.zeros((self.rom.n_robots,), device=self.device).bool()
        self.prob_rnd = prob_rnd
        self.rnd_inds = None
        self.rnd_inds_bool = torch.zeros((self.rom.n_robots,), device=self.device).bool()
        self.noise_max_std = torch.max(self.rom.v_max) * noise_max_std
        self.noise_llh = noise_llh

    def resample(self, idx, z):
        if len(idx) > 0:
            v_min, v_max = self.rom.compute_state_dependent_input_bounds(z[idx, :])
            self._resample_const_input(idx, v_min, v_max)
            self._resample_extreme_input(idx, v_min, v_max)
            self._resample_ramp_input(idx, z, v_min, v_max)
            self._resample_sinusoid_input(idx, v_min, v_max)
            self._resample_rnd_input(idx, v_min, v_max)
            self._resample_t_final(idx)
            self._resample_weight(idx)
            n = torch.sum(idx) if idx.dtype == bool else len(idx)
            self.stationary_inds[idx] = self.uniform(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device), (n, 1)).squeeze() < self.prob_stationary
            self.rnd_inds_bool[idx] = self.uniform(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device),
                                                     (n, 1)).squeeze() < self.prob_rnd
            self.rnd_inds = torch.nonzero(self.rnd_inds_bool).reshape(-1,)
            self.noise_std[idx] = self.uniform(torch.tensor([0.], device=self.device), self.noise_max_std, size=(len(idx), self.rom.m))
            mask = torch.rand(len(idx), device=self.device) > self.noise_llh
            self.noise_std[idx[mask]] = 0


    def _resample_t_final(self, idx):
        self.t_final[idx] += self.t_sampler.sample(len(idx))

    def _resample_weight(self, idx):
        self.weights[idx, :, :] = self.weight_sampler.sample(len(idx))

    def _resample_const_input(self, idx, v_min, v_max):
        self.sample_hold_input[idx, :] = self.uniform(v_min, v_max, size=(len(idx), self.rom.m))

    def _resample_ramp_input(self, idx, z, v_min, v_max):
        self.ramp_v_start[idx, :] = self.rom.clip_v_z(z[idx, :], self.ramp_v_end[idx, :])
        self.ramp_v_end[idx, :] = self.uniform(v_min, v_max, size=(len(idx), self.rom.m))
        self.ramp_t_start[idx] = self.t_final[idx]
        mask = torch.rand(len(idx), device=self.device) < 0.1
        self.ramp_v_end[idx[mask], :] = self.extreme_input[idx[mask], :]

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

    def _resample_rnd_input(self, idx, v_min, v_max):
        self.rnd_mag[idx, :] = self.uniform(torch.zeros_like(v_max), (v_max - v_min) / 2, size=(len(idx), self.rom.m))
        self.rnd_mean[idx, :] = self.uniform(v_min + self.sin_mag[idx, :], v_max - self.sin_mag[idx, :], size=(len(idx), self.rom.m))

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
        return self.weights[:, :, 0] * self.rom.clip_v_z(z, self._const_input()) + \
            self.weights[:, :, 1] * self.rom.clip_v_z(z, self._ramp_input_t(t)) + \
            self.weights[:, :, 2] * self.rom.clip_v_z(z, self._extreme_input()) + \
            self.weights[:, :, 3] * self.rom.clip_v_z(z, self._sinusoid_input_t(t))


    def step_rom_idx(self, idx, e_prev=None, increment_rom_time=False):
        # Get input to apply for trajectory
        v = self.get_input_t(self.t, self.trajectory[:, -1, :])
        v[self.stationary_inds, :] = 0
        # v_min, v_max = self.rom.compute_state_dependent_input_bounds(self.trajectory[self.rnd_inds, -1, :])
        v[self.rnd_inds, :] = self.uniform(-self.rnd_mag[self.rnd_inds, :], self.rnd_mag[self.rnd_inds, :], size=(len(self.rnd_inds), self.rom.m)) + self.rnd_mean[self.rnd_inds, :]
        self.v[idx, :] = self.rom.clip_v_z(self.trajectory[idx, -1, :],
            v[idx, :] + self.noise_std[idx, :] * torch.randn(len(idx), self.rom.m, device=self.device)
        )
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

class TrajectoryGeneratorH2H(TrajectoryGenerator):

    def __init__(self, rom, t_sampler, weight_sampler, dt_loop=0.02, N=4, freq_low=0.01, freq_high=10,
                 device='cuda', prob_stationary=.01, dN=1, prob_rnd=0.05, noise_max_std=0.1, noise_llh=0.25):
        super().__init__(
            rom, t_sampler, weight_sampler, dt_loop=dt_loop, N=N, freq_low=freq_low, freq_high=freq_high, device=device,
            prob_stationary=prob_stationary, dN=dN, prob_rnd=prob_rnd, noise_max_std=noise_max_std, noise_llh=noise_llh
        )
        self.in_contact = torch.zeros((self.rom.n_robots,), dtype=torch.bool, device=self.device)
        self.made_contact = torch.zeros((self.rom.n_robots,), dtype=torch.bool, device=self.device)

    def step_idx(self, idx, e_prev=None, dt_tol=1e-5):
        # first, remask for envs which need a rom step
        masked_idx = idx[self.made_contact[idx]]
        if len(masked_idx):
            self.step_rom_idx(masked_idx, e_prev=e_prev)
        self.t[idx] += self.dt_loop

    def reset_idx(self, idx, z, e_prev=None):
        self.made_contact[idx] = False
        self.in_contact[idx] = False
        super().reset_idx(idx, z, e_prev=e_prev)

    def update_made_contact(self, in_contact_k):
        self.made_contact = torch.logical_and(in_contact_k, torch.logical_not(self.in_contact))
        self.in_contact = torch.clone(in_contact_k.detach())


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

    def reset_idx(self, idx, z, e_prev=None):
        z[:, self.rom.vel_inds] = 0
        super().reset_idx(idx, z)
