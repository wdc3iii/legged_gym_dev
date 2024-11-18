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
from torch.utils.data import DataLoader
from scipy.io import savemat
from predictive_cbfs.utils import CheckPointManager, RegressionDataset


@hydra.main(
    config_path=str(Path(__file__).parent / "configs" / "delta_learning"),
    config_name="default_custom",
    version_base="1.2",
)
def data_creation_main(cfg):
    """____________________  Setup Learning Stuff  __________________________________________"""
    experiment_name = cfg.experiment_name

    # Send config to wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]


    import random
    import string
    total_run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    data_path = str(Path(__file__).parent / "predictive_cbfs" / f"{cfg.experiment_name}_{total_run_id}")
    os.makedirs(data_path, exist_ok=True)
    with open(f"{data_path}/config.pickle", "wb") as f:
        pickle.dump(cfg_dict, f)

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

    obs, _ = env.reset()
    x_n =obs.shape[1]
    eval_states = torch.clone(env.get_states().detach())
    eval_func = instantiate(cfg.eval_function)

    # Loop over epochs
    num_robots = env_cfg.env.num_envs
    max_ep_length = int(cfg.env_config.env.episode_length_s / env.dt) - 5


    """_____________________________________ Loop over Learning Iterations __________________________________________"""
    for ii in range(cfg.learning_iters):
        # Data structures
        x = torch.zeros((cfg.epochs, num_robots, max_ep_length + 1, x_n), device=env.device)  # Epochs, steps, states
        h = torch.zeros((cfg.epochs, num_robots, max_ep_length + 1), device=env.device)
        delta = torch.zeros((cfg.epochs, num_robots, max_ep_length + 1), device=env.device)


        """________________________________ Collect Data Under Current delta policy _________________________________"""
        for epoch in tqdm(range(cfg.epochs), desc="Data Collection Progress (epochs)"):

            # Initialization
            env.reset()
            obs = env.get_observations()
            x[epoch, :, 0, :] = obs.detach()
            h[epoch, :, 0] = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[epoch, :, 0, :]))

            # Loop over time steps
            for t in range(max_ep_length):
                # have to modify obs if using Raibert Heuristic
                actions = policy(obs.detach())
                obs, _, _, dones, _ = env.step(actions.detach())

                # Save Data
                x[epoch, :, t + 1, :] = obs.detach()
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
            savemat(f"{data_path}/data_{ii}.mat", epoch_data)


        """_____________________________________________ Learn a new delta policy ___________________________________"""
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
        err_h_bar = -sliding_min(h, int(cfg.p_cbf_horizon_s / env.dt))
        # Compute the new desired
        delta_targ = delta[:, :, :err_h_bar.shape[2]] + cfg.delta_K * err_h_bar
        delta_targ = torch.clip(delta_targ, 0, policy.v_filt.delta_max)


        # Reshape for learning
        x = x[:, :, :err_h_bar.shape[2], :]
        x = x.reshape(cfg.epochs * num_robots * x.shape[2], x.shape[3])
        delta_targ = delta_targ.reshape(cfg.epochs * num_robots * delta_targ.shape[2], 1)

        # Adjust points which are near origin?
        if cfg.adjust_distribution:
            # Example inputs

            # 1. Identify the proportion of rows close to the origin
            distances = torch.norm(x[:, :2], dim=1)  # Compute distances from origin
            close_mask = distances <= cfg.converged_tol  # Mask for rows within epsilon
            proportion_close = close_mask.float().mean().item()  # Proportion of rows close to origin
            far_mask = ~close_mask
            # 2. Randomly eliminate rows to ensure at most alpha percent are close
            num_close_to_keep = int(cfg.max_converged_proportion * x.shape[0])  # Maximum allowed rows close to origin
            close_indices = torch.where(close_mask)[0]  # Indices of close rows

            if len(close_indices) > num_close_to_keep:
                keep_indices = torch.randperm(len(close_indices))[:num_close_to_keep]
                close_indices_to_keep = close_indices[keep_indices]
                close_mask[close_indices] = False  # Reset all initially
                close_mask[close_indices_to_keep] = True  # Mark the subset to keep

                # Resulting matrix after elimination
                x = x[close_mask | far_mask]

        # Adjust points to keep
        if policy.keep_inds is not None:
            x = x[:, policy.keep_inds]

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
        ckpt_manager.to_wandb()
        wandb.finish()

        # copy the best NN over
        checkpoint_path = f"{ckpt_manager.ckpt_path}/best_model.pth"
        state_dict = torch.load(checkpoint_path)
        # new_delta.load_state_dict(state_dict)
        # policy.v_filt.delta = new_delta
        policy.v_filt.delta.load_state_dict(state_dict)

        # Evaluate new NN?
        eval_func(policy, env, eval_states, cfg, data_path, ii)

    print(f"\nrun ID: {total_run_id}\ndataset name: {cfg.experiment_name}\nlocal folder: {cfg.experiment_name}_{total_run_id}")
    return epoch_data


if __name__ == "__main__":
    data_creation_main()
