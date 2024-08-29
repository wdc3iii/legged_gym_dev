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
import matplotlib.pyplot as plt
from isaacgym.torch_utils import *
from deep_tube_learning.utils import update_args_from_hydra, update_cfgs_from_hydra, wandb_model_load, update_hydra_cfg
from deep_tube_learning.controllers import RaibertHeuristic
from trajopt.rom_dynamics import SingleInt2D
from deep_tube_learning.custom_sim import CustomSim


@hydra.main(
    config_path=str(Path(__file__).parent / "configs" / "data_generation"),
    config_name="default_trajectory",
    version_base="1.2",
)
def data_creation_main(cfg):
    experiment_name = cfg.dataset_name

    # Send config to wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]
    if cfg.upload_to_wandb:
        wandb.init(project="RoM_Tracking_Data",
                   entity="coleonguard-Georgia Institute of Technology",
                   name=experiment_name,  # Use the dynamic experiment name
                   config=cfg_dict)
        run_id = wandb.run.id
    else:
        import random
        import string
        run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    data_path = str(Path(__file__).parent / "rom_tracking_data" / f"{cfg.dataset_name}_{run_id}")
    os.makedirs(data_path, exist_ok=True)

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
        policy = instantiate(cfg.controller)(state_dependent_input_bound=env.model.clip_v_z)
    else:
        raise ValueError(f"Environment type {cfg.env_config.env.type} not implemented.")


    obs = env.get_observations()
    x_n = env.get_states().shape[1]

    # Loop over epochs
    num_robots = env_cfg.env.num_envs
    rom_n = env.rom.n
    rom_m = env.rom.m
    max_rom_ep_length = int(cfg.env_config.env.episode_length_s / env.rom.dt)
    for epoch in tqdm(range(cfg.epochs), desc="Data Collection Progress (epochs)"):
        # Data structures
        x = torch.zeros((num_robots, max_rom_ep_length + 1, x_n), device=env.device)  # Epochs, steps, states
        z = torch.zeros((num_robots, max_rom_ep_length + 1, rom_n), device=env.device)
        pz_x = torch.zeros((num_robots, max_rom_ep_length + 1, rom_n), device=env.device)
        v = torch.zeros((num_robots, max_rom_ep_length, rom_m), device=env.device)
        done = torch.zeros((num_robots, max_rom_ep_length), dtype=torch.bool, device=env.device)

        # Initialization
        env.reset()
        base = torch.clone(env.root_states.detach())
        x[:, 0, :] = env.get_states()
        pz_x[:, 0, :] = env.rom.proj_z(base)

        z[:, 0, :] = env.traj_gen.trajectory[:, 0, :]

        # Loop over time steps
        for t in range(max_rom_ep_length):
            # Step environment until rom steps
            k = torch.clone(env.traj_gen.k.detach())
            while torch.any(env.traj_gen.k == k):
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
            base = torch.clone(env.root_states.detach())
            d = torch.clone(dones.detach())
            proj = env.rom.proj_z(base)
            done[:, t] = d  # Termination should not be used for tube training
            v[:, t, :] = env.traj_gen.v
            x[:, t + 1, :] = env.get_states()
            z[:, t + 1, :] = env.traj_gen.get_trajectory()[:, 0, :]
            z[done[:, t], t + 1, :] = proj[done[:, t], :]  # Terminated envs reset to zero tracking error
            pz_x[:, t + 1, :] = proj

        # Plot the trajectories after the loop
        plt.figure()
        plt.plot(v[0, :, :].cpu().numpy())
        plt.show()
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(z[0, :, :].cpu().numpy())
        ax[0].plot(pz_x[0, :, :].cpu().numpy())
        env.rom.plot_spacial(ax[1], z[0, :, :].cpu().numpy(), '.-k')
        env.rom.plot_spacial(ax[1], pz_x[0, :, :].cpu().numpy())
        plt.show()


        # Log Data
        with open(f"{data_path}/epoch_{epoch}.pickle", "wb") as f:
            if cfg.save_debugging_data:
                epoch_data = {
                    'x': x.cpu().numpy(),
                    'z': z.cpu().numpy(),
                    'v': v.cpu().numpy(),
                    'pz_x': pz_x.cpu().numpy(),
                    'done': done.cpu().numpy()
                }
            else:
                epoch_data = {
                    'z': z.cpu().numpy(),
                    'v': v.cpu().numpy(),
                    'pz_x': pz_x.cpu().numpy(),
                    'done': done.cpu().numpy()
                }
            pickle.dump(epoch_data, f)

    if cfg.upload_to_wandb:
        artifact = wandb.Artifact(
            type="rom_tracking_data",
            name=f"{cfg.dataset_name}_{wandb.run.id}"
        )
        artifact.add_dir(str(data_path))
        wandb.run.log_artifact(artifact)
        print("[INFO] Finished generating data. Waiting for wandb to finish uploading...")

    print(f"\nrun ID: {run_id}\ndataset name: {cfg.dataset_name}\nlocal folder: {cfg.dataset_name}_{run_id}")
    return epoch_data


if __name__ == "__main__":
    data_creation_main()
