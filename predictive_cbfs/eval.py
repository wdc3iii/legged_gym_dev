import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import pickle
from tqdm import tqdm
from pathlib import Path
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from isaacgym.torch_utils import *
from deep_tube_learning.utils import update_args_from_hydra, update_cfgs_from_hydra, update_hydra_cfg
from predictive_cbfs.custom_sim import CustomSim
from scipy.io import savemat

from deep_tube_learning.utils import unnormalize_dict


dir_name = "double_single_int_j4fo3cmm"

def main():
    data_path = str(Path(__file__).parent / "predictive_cbfs" / dir_name)
    with open(f"{data_path}/config.pickle", 'rb') as file:
        cfg_dict = pickle.load(file)

    cfg_dict['env_config/env/num_envs'] = 1
    cfg_dict['env_config/env/episode_length_s'] = 10

    cfg = OmegaConf.create(unnormalize_dict(cfg_dict))

    if cfg.env_config.env.type == 'isaacgym':
        cfg_dir = str(Path(__file__).parent.resolve() / "configs" / "isaac")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.2"):
            rl_cfg = compose(config_name="hopper")
        rl_cfg = update_hydra_cfg(cfg, rl_cfg)

        args = get_args()
        args = update_args_from_hydra(rl_cfg, args)
        env_cfg, train_cfg = task_registry.get_cfgs(rl_cfg.task)
        env_cfg, train_cfg = update_cfgs_from_hydra(rl_cfg, env_cfg, train_cfg)

        env, env_cfg = task_registry.make_env(name=rl_cfg.task, args=args, env_cfg=env_cfg)

        policy = instantiate(cfg.controller)
    elif cfg.env_config.env.type == 'custom':
        env_cfg = cfg.env_config
        env = CustomSim(env_cfg)
        policy = instantiate(cfg.controller)
    else:
        raise ValueError(f"Environment type {cfg.env_config.env.type} not implemented.")

    checkpoint_path = f"models/{dir_name[-8:]}/best_model.pth"
    state_dict = torch.load(checkpoint_path, weights_only=True)
    policy.v_filt.delta.load_state_dict(state_dict)

    env.set_states(torch.tensor([[-3., -5., 0., 0.]], device="cuda:0"))
    obs = env.get_observations()
    x_n = obs.shape[1]

    max_ep_length = int(cfg.env_config.env.episode_length_s / env.dt)

    # Data structures
    x = torch.zeros((max_ep_length + 1, x_n), device=env.device)  # Epochs, steps, states
    vd = torch.zeros((max_ep_length + 1, 2), device=env.device)  # Epochs, steps, states
    v = torch.zeros((max_ep_length + 1, 2), device=env.device)  # Epochs, steps, states
    h = torch.zeros((max_ep_length + 1), device=env.device)
    delta = torch.zeros((max_ep_length + 1), device=env.device)


    # Initialization
    x[0, :] = obs.detach().squeeze(0)
    vd[0, :] = policy.v_filt.vd(policy.v_filt.dyn.proj(obs.detach())).squeeze(0)
    v[0, :] = policy.v_filt(obs.detach()).squeeze(0)
    h0 = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[None, 0, :]))
    h[0] = h0

    # Loop over time steps
    for t in range(max_ep_length):
        # have to modify obs if using Raibert Heuristic
        actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        # Save Data
        x[t + 1, :] = obs.detach().squeeze(0)
        vd[t + 1, :] = policy.v_filt.vd(policy.v_filt.dyn.proj(obs.detach())).squeeze(0)
        v[t + 1, :] = policy.v_filt(obs.detach()).squeeze(0)
        ht = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[None, t + 1, :]))
        h[t + 1] = ht.squeeze(0)
        delta[t + 1] = policy.v_filt.delta_val.detach().squeeze(0)

    savemat(
        f"{data_path}/eval_data.mat",
        {"x": x.cpu().numpy(), "h": h.cpu().numpy(), "delta": delta.cpu().numpy(), "v": v.cpu().numpy(), "vd": vd.cpu().numpy()}
    )

if __name__ == "__main__":
    main()