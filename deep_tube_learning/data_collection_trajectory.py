import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import hydra
import torch
import wandb
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from hydra.utils import instantiate

from deep_tube_learning.utils import quat2yaw, yaw2rot, wrap_angles


CMD_START_IDX = 9


def get_state(base, joint_pos, joint_vel):
    return np.concatenate((base[:, :7], joint_pos, base[:, 7:], joint_vel), axis=1)


@hydra.main(
    config_path=str(Path(__file__).parent / "configs" / "data_generation"),
    config_name="default",
    version_base="1.2",
)
def data_creation_main(cfg):
    # Seed RNG
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

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

    # Load configuration
    num_robots = cfg.num_robots
    rom = instantiate(cfg.reduced_order_model)
    traj_gen = instantiate(cfg.trajectory_generator)(rom=rom)
    track_yaw = cfg.track_yaw

    env_cfg, train_cfg = task_registry.get_cfgs(name=cfg.task)

    # Override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, num_robots)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # Prepare environment
    args = get_args()
    args.headless = cfg.headless
    env, _ = task_registry.make_env(name=cfg.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    x_n = env.dof_pos.shape[1] + env.dof_vel.shape[1] + env.root_states.shape[1]

    CMD = np.arange(CMD_START_IDX, CMD_START_IDX + env.traj_gen.N * env.rom.n)

    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cfg.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Loop over epochs
    for epoch in tqdm(range(cfg.epochs), desc="Data Collection Progress (epochs)"):
        # Data structures
        x = np.zeros((int(env.max_episode_length) + 1, num_robots, x_n))  # Epochs, steps, states
        z = np.zeros((int(env.max_episode_length) + 1, num_robots, rom.n))
        pz_x = np.zeros((int(env.max_episode_length) + 1, num_robots, rom.n))
        v = np.zeros((int(env.max_episode_length), num_robots, rom.m))
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
        z[0, :, :] = rom.proj_z(base)
        pz_x[0, :, :] = rom.proj_z(base)

        traj_gen.reset(rom.proj_z(env.root_states).cpu())

        # Loop over time steps
        for t in range(int(env.max_episode_length)):
            # Get desired pose and velocity
            traj = torch.clone(torch.from_numpy(traj_gen.trajectory).to(env.device).float())
            traj -= rom.proj_z(env.root_states)[:, None, :2]

            des_pose, des_vel = rom.des_pose_vel(z[t, :, :], traj_gen.v)
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

            obs[:, CMD] = traj.reshape(env.num_envs, -1)
            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())

            # Save Data
            base = env.root_states.cpu().numpy()
            d = dones.cpu().numpy()
            proj = rom.proj_z(base)
            traj_gen.reset_idx(np.nonzero(d)[0], proj)
            done[t, :] = d  # Termination should not be used for tube training
            v[t, :, :] = traj_gen.v
            x[t + 1, :, :] = get_state(base, env.dof_pos.cpu().numpy(), env.dof_vel.cpu().numpy())
            z[t + 1, :, :] = traj_gen.trajectory[:, 0, :]
            z[t + 1, done[t, :], :] = proj[done[t, :], :]  # Terminated envs reset to zero tracking error
            pz_x[t + 1, :, :] = rom.proj_z(base)

            # print(np.linalg.norm(pz_x[t, :, :] - z[t, :, :], axis=-1))

            traj_gen.step()

        # Plot the trajectories after the loop
        fig, ax = plt.subplots()
        rom.plot_spacial(ax, z[:, 0, :], '.-k')
        rom.plot_spacial(ax, pz_x[:, 0, :], '.-b')
        plt.show()
        fig, ax = plt.subplots()
        rom.plot_spacial(ax, z[:, 1, :], '.-k')
        rom.plot_spacial(ax, pz_x[:, 1, :], '.-b')
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
