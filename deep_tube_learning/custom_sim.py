from trajopt.rom_dynamics import SingleInt2D, DoubleInt2D, TrajectoryGenerator
from deep_tube_learning.utils import *


class CustomSim:

    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = self.cfg.env.model.dt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_envs = self.cfg.env.num_envs

        model_cfg = self.cfg.env.model
        model_class = globals()[model_cfg.cls]
        self.model = model_class(
            dt=self.dt,
            z_min=torch.tensor(model_cfg.z_min, device=self.device),
            z_max=torch.tensor(model_cfg.z_max, device=self.device),
            v_min=torch.tensor(model_cfg.v_min, device=self.device),
            v_max=torch.tensor(model_cfg.v_max, device=self.device),
            n_robots=self.num_envs,
            backend='torch',
            device=self.device
        )

        self._init_rom()
        self._init_trajectory_generator()

        self.root_states = torch.zeros((self.num_envs, self.model.n), device=self.device)
        self.trajectory = torch.zeros(self.num_envs, self.traj_gen.N, self.rom.n, dtype=torch.float, device=self.device,
                                      requires_grad=False)
        self.max_rom_distance = torch.tensor(self.cfg.domain_rand.max_rom_distance, device=self.device)
        self.zero_rom_dist_llh = self.cfg.domain_rand.zero_rom_dist_llh
        self.root_state_noise_lower = torch.tensor(self.cfg.init_state.default_noise_lower, device=self.device)
        self.root_state_noise_upper = torch.tensor(self.cfg.init_state.default_noise_upper, device=self.device)

    def _init_rom(self):
        rom_cfg = self.cfg.rom
        model_class = globals()[rom_cfg.cls]
        self.rom = model_class(
            dt=rom_cfg.dt,
            z_min=torch.tensor(rom_cfg.z_min, device=self.device),
            z_max=torch.tensor(rom_cfg.z_max, device=self.device),
            v_min=torch.tensor(rom_cfg.v_min, device=self.device),
            v_max=torch.tensor(rom_cfg.v_max, device=self.device),
            n_robots=self.cfg.env.num_envs,
            backend='torch',
            device=self.device
        )

    def _init_trajectory_generator(self):
        traj_cfg = self.cfg.trajectory_generator
        traj_cls = globals()[traj_cfg.cls]
        t_samp = globals()[traj_cfg.t_samp_cls](traj_cfg.t_low, traj_cfg.t_high, backend='torch', device=self.device)
        weight_samp = globals()[traj_cfg.weight_samp_cls]()
        self.traj_gen = traj_cls(
            self.rom,
            t_samp,
            weight_samp,
            dt_loop=self.dt,
            N=traj_cfg.N,
            freq_low=traj_cfg.freq_low,
            freq_high=traj_cfg.freq_high,
            seed=traj_cfg.seed,
            backend='torch',
            device=self.device,
            prob_stationary=traj_cfg.prob_stationary,
            dN=traj_cfg.dN,
        )

    def step(self, action):
        self.root_states = self.model.f(self.root_states, action)
        self.traj_gen.step()
        self.trajectory = torch.clone(self.traj_gen.get_trajectory().detach())
        return self.get_observations(), None, None, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), None

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def reset_traj(self, env_ids):
        p_zx = self.rom.proj_z(torch.clone(self.root_states.detach()))
        if self.cfg.domain_rand.randomize_rom_distance:
            mask = torch.rand(len(env_ids)) > self.zero_rom_dist_llh
            p_zx[env_ids[mask], :] += torch_rand_vec_float(-self.max_rom_distance, self.max_rom_distance, p_zx[env_ids[mask], :].shape, device=self.device)
        self.traj_gen.reset_idx(env_ids, p_zx)

    def reset_idx(self, idx):
        self.root_states[idx, :] = torch_rand_vec_float(
            self.root_state_noise_lower, self.root_state_noise_upper,
            (len(idx), self.model.n), device=self.device
        )
        self.reset_traj(idx)
        self.step(torch.zeros(self.num_envs, self.model.m, device=self.device))

    def get_observations(self):
        return torch.concatenate((
            torch.clone(self.root_states.detach()),
            self.trajectory[:, 0, :],
            torch.clone(self.traj_gen.v_trajectory[:, 1, :].detach())
        ), dim=1)

    def get_states(self):
        return torch.clone(self.root_states.detach())