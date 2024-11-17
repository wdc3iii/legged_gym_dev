from deep_tube_learning.utils import torch_rand_vec_float
import torch
from predictive_cbfs.roms import DoubleInt2D, DoubleInt1D


class CustomSim:

    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = self.cfg.env.model.dt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_envs = self.cfg.env.num_envs

        model_cfg = self.cfg.env.model
        if model_cfg.cls == 'DoubleInt2D':
            self.model = DoubleInt2D(model_cfg.dt)
        elif model_cfg.cls == 'DoubleInt1D':
            self.model = DoubleInt1D(model_cfg.dt)
        else:
            raise ValueError(f"Model {model_cfg.cls} not supported")

        self.root_states = torch.zeros((self.num_envs, self.model.n), device=self.device)
        self.root_state_noise_lower = torch.tensor(self.cfg.init_state.default_noise_lower, device=self.device)
        self.root_state_noise_upper = torch.tensor(self.cfg.init_state.default_noise_upper, device=self.device)

    def step(self, action):
        self.root_states = self.model(self.root_states, action)
        return self.get_observations(), None, None, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), None

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def reset_idx(self, idx):
        self.root_states[idx, :] = torch_rand_vec_float(
            self.root_state_noise_lower, self.root_state_noise_upper,
            (len(idx), self.model.n), device=self.device
        )

    def get_observations(self):
        return torch.clone(self.root_states.detach())

    def get_states(self):
        return torch.clone(self.root_states.detach())