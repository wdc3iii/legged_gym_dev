import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import os
import hydra
import wandb
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from isaacgym.torch_utils import *
from deep_tube_learning.utils import update_args_from_hydra, update_cfgs_from_hydra, wandb_model_load, update_hydra_cfg
from deep_tube_learning.controllers import RaibertHeuristic
from trajopt.rom_dynamics import SingleInt2D
from predictive_cbfs.custom_sim import CustomSim
from deep_tube_learning.train_tube import CheckPointManager
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@hydra.main(
    config_path=str(Path(__file__).parent / "configs" / "data_generation"),
    config_name="default_trajectory",
    version_base="1.2",
)
def data_creation_main(cfg):
    """____________________  Setup Learning Stuff  __________________________________________"""
    experiment_name = cfg.dataset_name

    # Send config to wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]


    import random
    import string
    total_run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    data_path = str(Path(__file__).parent / "rom_tracking_data" / f"{cfg.dataset_name}_{total_run_id}")
    os.makedirs(data_path, exist_ok=True)
    with open(f"{data_path}/config.pickle", "wb") as f:
        pickle.dump(cfg_dict, f)

    if cfg.env_config.env.type == 'isaacgym':
        if cfg.controller.type == 'rl':
            exp_name = cfg.wandb_experiment
            model_name = f'{exp_name}_model:best{cfg.curriculum}'
            api = wandb.Api()
            rl_cfg, state_dict = wandb_model_load(api, model_name)
        elif cfg.controller.type == 'rh':
            cfg_dir = str(Path(__file__).parent / "configs" / "rl")
            GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=cfg_dir, version_base="1.2"):
                rl_cfg = compose(config_name=cfg.controller.config_name)
        else:
            raise ValueError(f"Controller type {cfg.controller.type} not implemented.")
        rl_cfg = update_hydra_cfg(cfg, rl_cfg)

        args = get_args()
        args = update_args_from_hydra(rl_cfg, args)
        env_cfg, train_cfg = task_registry.get_cfgs(rl_cfg.task)
        env_cfg, train_cfg = update_cfgs_from_hydra(rl_cfg, env_cfg, train_cfg)

        env, env_cfg = task_registry.make_env(name=rl_cfg.task, args=args, env_cfg=env_cfg)

        if cfg.controller.type == 'rl':
            train_cfg.runner.resume = True
            ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                                  train_cfg=train_cfg)
            policy = ppo_runner.get_inference_policy(device=env.device)
        elif cfg.controller.type == 'rh':
            raibert = RaibertHeuristic(cfg)
            policy = raibert.get_inference_policy(device=env.device)
        else:
            raise ValueError(f"IsaacGym controller type {cfg.controller.type} not implemented.")
    elif cfg.env_config.env.type == 'custom':
        env_cfg = cfg.env_config
        env = CustomSim(env_cfg)
        policy = instantiate(cfg.controller)
    else:
        raise ValueError(f"Environment type {cfg.env_config.env.type} not implemented.")

    x_n = env.get_states().shape[1]
    env.reset()
    eval_root_states = torch.clone(env.root_states.detach())
    eval_func = instantiate(cfg.eval_function)

    # Loop over epochs
    num_robots = env_cfg.env.num_envs
    max_rom_ep_length = int(cfg.env_config.env.episode_length_s / env.model.dt) - 5

    """_____________________________________ Loop over Learning Iterations __________________________________________"""
    for ii in range(cfg.learning_iters):
        # Data structures
        x = torch.zeros((cfg.epochs, num_robots, max_rom_ep_length + 1, x_n), device=env.device)  # Epochs, steps, states
        h = torch.zeros((cfg.epochs, num_robots, max_rom_ep_length + 1), device=env.device)
        delta = torch.zeros((cfg.epochs, num_robots, max_rom_ep_length + 1), device=env.device)

        """________________________________ Collect Data Under Current delta policy _________________________________"""
        for epoch in tqdm(range(cfg.epochs), desc="Data Collection Progress (epochs)"):

            # Initialization
            env.reset()
            obs = env.get_observations()
            x[epoch, :, 0, :] = env.get_states()
            h[epoch, :, 0] = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[epoch, :, 0, :]))

            # Loop over time steps
            for t in range(max_rom_ep_length):
                # have to modify obs if using Raibert Heuristic
                if 'type' in cfg.controller.keys() and cfg.controller.type == 'rh':
                    if isinstance(env.traj_gen.rom, SingleInt2D):
                        current_velocity = env.root_states[:, 7:9]
                        current_position = env.root_states[:, :2]

                        desired_position = env.traj_gen.get_trajectory()[:, cfg.controller.N]
                        desired_velocity = env.traj_gen.get_v_trajectory()[:, cfg.controller.N]

                        positional_error = desired_position - current_position
                        # velocity_error = desired_velocity - current_velocity
                        quaternion = env.base_quat  # x,y,z,w
                        obs = torch.cat((positional_error, current_velocity, desired_velocity, quaternion), dim=1)
                actions = policy(obs.detach())
                obs, _, _, dones, _ = env.step(actions.detach())

                # Save Data
                x[epoch, :, t + 1, :] = env.get_states()
                h[epoch, :, t + 1] = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[epoch, :, t+1, :]))
                delta[epoch, :, t + 1] = policy.v_filt.delta_val.detach()

        # Log Data
        with open(f"{data_path}/data_{ii}.pickle", "wb") as f:
            epoch_data = {
                'x': x.cpu().numpy(),
                'h': h.cpu().numpy(),
                'delta': delta.cpu().numpy(),
            }
            pickle.dump(epoch_data, f)

        """_____________________________________________ Learn a new delta policy ___________________________________"""
        print("Time to learn...")

        def sliding_min(h_, T):
            n1, n2, H = h.shape

            strides = h_.stride()
            sliding_windows = torch.as_strided(
                h_,
                size=(n1, n2, H-T+1, T),
                stride=(strides[0], strides[1], strides[2], strides[2])
            )
            return torch.min(sliding_windows, dim=-1).values

        # Compute minimum violation over the horizon
        err_h_bar = -sliding_min(h, int(cfg.p_cbf_horizon_s / env.model.dt))
        # Compute the new desired
        delta_targ = delta[:, :, :err_h_bar.shape[2]] + cfg.delta_K * err_h_bar
        delta_targ = torch.clip(delta_targ, 0, cfg.delta_max)


        # Reshape for learning
        x = x[:, :, :err_h_bar.shape[2], :]
        x = x.reshape(cfg.epochs * num_robots * x.shape[2], x.shape[3])
        delta_targ = delta_targ.reshape(cfg.epochs * num_robots * delta_targ.shape[2], 1)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(x[::100, 0].cpu().numpy(), x[::100, 1].cpu().numpy(), c=delta_targ[::100].cpu().numpy())
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.colorbar()
        # plt.title('Delta Target')
        # plt.show()

        # Downsample to avoid temporal corrolation
        n_samples = int(x.shape[0] * cfg.decorr_prop)
        inds = torch.randperm(x.shape[0])[:n_samples]
        x = x[inds, :]
        delta_targ = delta_targ[inds]
        loader = DataLoader(RegressionDataset(x, delta_targ), batch_size=cfg.batch_size, shuffle=True)

        # Get a new NN
        new_delta = instantiate(cfg.controller.robustified_cbf.delta).to(env.device)
        optimizer = instantiate(cfg.optimizer)(new_delta.parameters())
        lr_scheduler = instantiate(cfg.lr_scheduler)(optimizer)
        loss_fn = instantiate(cfg.loss_fn)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]
        wandb.init(project="Predictive_CBFs",
                   entity="wdc3iii",
                   name=f"{experiment_name}_{ii}",  # Use the dynamic experiment name
                   config=cfg_dict)

        ckpt_manager = CheckPointManager(metric_name="loss")
        # Fit the nn
        step = 0
        for t_ep in range(cfg.train_epochs):
            new_delta.train()
            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {t_ep}/{cfg.train_epochs}")
            for batch in pbar:
                step += 1
                data, targets = batch[0].to(env.device), batch[1].to(env.device)
                optimizer.zero_grad()
                outputs = new_delta(data)
                loss = loss_fn(outputs, targets)

                loss.backward()
                if 'grad_clip' in cfg.keys():
                    clip_grad_norm_(new_delta.parameters(), cfg.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                epoch_loss += loss.item()

                # Compute gradient norm
                grads = [
                    param.grad.detach().flatten()
                    for param in new_delta.parameters()
                    if param.grad is not None
                ]
                grad_norm = torch.cat(grads).norm()

                # Log loss, lr, and gradient norm
                wandb.log(
                    {
                        "loss_step": loss.item(),
                        "lr_step": lr_scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm
                    },
                    step=step,
                )
                pbar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
                epoch_loss += loss
                if step % cfg.steps_per_model_checkpoint == 0:
                    ckpt_manager.save(new_delta, loss.item(), epoch=t_ep, step=step)

            wandb.log(
                {"loss_epoch": epoch_loss.item() / len(loader), "lr_epoch": lr_scheduler.get_last_lr()[0]},
                step=step,
            )
        wandb.finish()

        # copy the new NN over
        policy.v_filt.delta = new_delta

        # Evaluate new NN?
        eval_func(policy, env, eval_root_states, cfg, data_path, ii)

    print(f"\nrun ID: {total_run_id}\ndataset name: {cfg.dataset_name}\nlocal folder: {cfg.experiment_name}_{total_run_id}")
    return epoch_data


if __name__ == "__main__":
    data_creation_main()
