import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import os
import hydra
import wandb
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from deep_tube_learning.utils import (update_args_from_hydra, update_cfgs_from_hydra, quat2yaw, yaw2rot, wrap_angles,
                                      wandb_model_load, update_hydra_cfg)


def get_state(base, joint_pos, joint_vel):
    return np.concatenate((base[:, :7], joint_pos, base[:, 7:], joint_vel), axis=1)


@hydra.main(
    config_path=str(Path(__file__).parent / "configs" / "data_generation"),
    config_name="default_trajectory",
    version_base="1.2",
)
def data_creation_main(cfg):

    exp_name = cfg.wandb_experiment
    model_name = f'{exp_name}_model:best'
    api = wandb.Api()
    rl_cfg, state_dict = wandb_model_load(api, model_name)
    rl_cfg = update_hydra_cfg(cfg, rl_cfg)

    # Send config to wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]
    if cfg.upload_to_wandb:
        wandb.init(project="RoM_Tracking_Data",
                   entity="coleonguard-Georgia Institute of Technology",
                   name=cfg.dataset_name,
                   config=cfg_dict)
        run_id = wandb.run.id
    else:
        import random
        import string
        run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    data_path = str(Path(__file__).parent / "rom_tracking_data" / f"{cfg.dataset_name}_{run_id}")
    os.makedirs(data_path, exist_ok=True)

    args = get_args()
    args = update_args_from_hydra(rl_cfg, args)
    env_cfg, train_cfg = task_registry.get_cfgs(rl_cfg.task)
    env_cfg, train_cfg = update_cfgs_from_hydra(rl_cfg, env_cfg, train_cfg)

    env, env_cfg = task_registry.make_env(name=rl_cfg.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    x_n = env.dof_pos.shape[1] + env.dof_vel.shape[1] + env.root_states.shape[1]

    # Loop over epochs
    num_robots = env_cfg.env.num_envs
    rom_n = env.rom.n
    rom_m = env.rom.m
    track_yaw = cfg.track_yaw
    for epoch in tqdm(range(cfg.epochs), desc="Data Collection Progress (epochs)"):
        # Data structures
        x = np.zeros((int(env.max_episode_length) + 1, num_robots, x_n))  # Epochs, steps, states
        z = np.zeros((int(env.max_episode_length) + 1, num_robots, rom_n))
        pz_x = np.zeros((int(env.max_episode_length) + 1, num_robots, rom_n))
        v = np.zeros((int(env.max_episode_length), num_robots, rom_m))
        done = np.zeros((int(env.max_episode_length), num_robots), dtype='bool')
        des_pose_all = np.zeros((int(env.max_episode_length), num_robots, 3))
        des_vel_all = np.zeros((int(env.max_episode_length), num_robots, 3))
        des_vel_local_all = np.zeros((int(env.max_episode_length), num_robots, 3))
        robot_pose_all = np.zeros((int(env.max_episode_length), num_robots, 3))
        robot_vel_all = np.zeros((int(env.max_episode_length), num_robots, 3))
        err_global_all = np.zeros((int(env.max_episode_length), num_robots, 3))
        err_local_all = np.zeros((int(env.max_episode_length), num_robots, 3))

        # Initialization
        base = env.root_states.cpu().numpy()
        x[0, :, :] = get_state(base, env.dof_pos.cpu().numpy(), env.dof_vel.cpu().numpy())
        z[0, :, :] = env.rom.proj_z(base)
        pz_x[0, :, :] = env.rom.proj_z(base)

        env.traj_gen.reset(env.rom.proj_z(env.root_states).cpu())

        # Loop over time steps
        for t in range(int(env.max_episode_length)):
            # Get desired pose and velocity

            des_pose, des_vel = env.rom.des_pose_vel(z[t, :, :], env.traj_gen.v)
            des_pose_all[t, :, :] = des_pose.copy()
            des_vel_all[t, :, :] = des_vel.copy()

            # Get robot pose
            robot_pose = np.hstack((base[:, :2], quat2yaw(base[:, 3:7])[:, None]))
            robot_pose_all[t, :, :] = robot_pose.copy()
            robot_vel_all[t, :, :] = np.hstack((base[:, 7:9], base[:, -1][:, None]))
            y2r = yaw2rot(robot_pose[:, 2])

            # Compute pose error
            err_global = des_pose - robot_pose
            err_global[:, 2] = np.where(track_yaw, wrap_angles(err_global[:, 2]), 0)
            err_global_all[t, :, :] = err_global.copy()
            err_local = err_global.copy()
            err_local[:, :2] = np.squeeze(y2r @ err_global[:, :2][:, :, None])  # Place error in local frame
            err_local_all[t, :, :] = err_local.copy()

            # Step environment
            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())

            # Save Data
            base = env.root_states.cpu().numpy()
            d = dones.cpu().numpy()
            proj = env.rom.proj_z(base)
            done[t, :] = d  # Termination should not be used for tube training
            v[t, :, :] = env.traj_gen.v
            x[t + 1, :, :] = get_state(base, env.dof_pos.cpu().numpy(), env.dof_vel.cpu().numpy())
            z[t + 1, :, :] = env.traj_gen.trajectory[:, 0, :]
            z[t + 1, done[t, :], :] = proj[done[t, :], :]  # Terminated envs reset to zero tracking error
            pz_x[t + 1, :, :] = env.rom.proj_z(base)

        # Plot the trajectories after the loop
        fig, ax = plt.subplots()
        env.rom.plot_spacial(ax, z[:, 0, :], '.-k')
        env.rom.plot_spacial(ax, pz_x[:, 0, :], '.-b')
        plt.show()
        fig, ax = plt.subplots()
        env.rom.plot_spacial(ax, z[:, 1, :], '.-k')
        env.rom.plot_spacial(ax, pz_x[:, 1, :], '.-b')
        plt.show()
        # Log Data
        with open(f"{data_path}/epoch_{epoch}.pickle", "wb") as f:
            if cfg.save_debugging_data:
                epoch_data = {
                    'x': x,
                    'z': z,
                    'v': v,
                    'pz_x': pz_x,
                    'done': done,
                    'des_pose': des_pose_all,
                    'des_vel': des_vel_all,
                    'des_vel_local': des_vel_local_all,
                    'robot_pose': robot_pose_all,
                    'robot_vel': robot_vel_all,
                    'err_local': err_local_all,
                    'err_global': err_global_all
                }
            else:
                epoch_data = {
                    'z': z,
                    'v': v,
                    'pz_x': pz_x,
                    'done': done
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
