import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import hydra
import torch
import wandb
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

from deep_tube_learning.utils import quat2yaw, yaw2rot


CMD_START_IDX = 9
CMD_LIN_VEL_PAR_IDX = CMD_START_IDX
CMD_LIN_VEL_PERP_IDX = CMD_START_IDX + 1
CMD_ANG_VEL_YAW_IDX = CMD_START_IDX + 2
CMD = np.array([CMD_LIN_VEL_PAR_IDX, CMD_LIN_VEL_PERP_IDX, CMD_ANG_VEL_YAW_IDX])


def get_state(base, joint_pos, joint_vel):
    return np.concatenate((base[:, :7], joint_pos, base[:, 7:], joint_vel), axis=1)


@hydra.main(
    config_path=str(Path(__file__).parent / "configs" / "data_generation"),
    config_name="default",
    version_base="1.2",
)
def data_creation_main(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]
    wandb.init(project="RoM_Tracking_Data",
               # entity="=coleonguard-Georgia%20Institute%20of%20Technology",
               name=cfg.dataset_name,
               config=cfg_dict)
    data_path = str(Path(__file__).parent / "rom_tracking_data" / f"{wandb.run.id}")
    os.makedirs(data_path, exist_ok=True)

    # Load configuration
    num_robots = cfg.num_robots
    rom = instantiate(cfg.reduced_order_model)
    sample_hold_dt = instantiate(cfg.sample_hold_dt)
    Kp = instantiate(cfg.Kp)
    track_yaw = cfg.track_yaw
    u_min = instantiate(cfg.u_min)
    u_max = instantiate(cfg.u_max)

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
    env, _ = task_registry.make_env(name=cfg.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    x_n = env.dof_pos.shape[1] + env.dof_vel.shape[1] + env.root_states.shape[1]

    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cfg.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Loop over epochs
    for epoch in range(cfg.epochs):
        # Data structures
        x = np.zeros((int(env.max_episode_length) + 1, num_robots, x_n))  # Epochs, steps, states
        u = np.zeros((int(env.max_episode_length), num_robots, 3))
        z = np.zeros((int(env.max_episode_length) + 1, num_robots, rom.n))
        pz_x = np.zeros((int(env.max_episode_length) + 1, num_robots, rom.n))
        v = np.zeros((int(env.max_episode_length), num_robots, rom.m))
        done = np.zeros((int(env.max_episode_length) + 1, num_robots), dtype='bool')

        # Initialization
        base = env.root_states.cpu().numpy()
        x[0, :, :] = get_state(base, env.dof_pos.cpu().numpy(), env.dof_vel.cpu().numpy())
        z[0, :, :] = rom.proj_z(base)
        pz_x[0, :, :] = rom.proj_z(base)

        v_nom = rom.sample_uniform_bounded_v(z[0, :, :])
        t_since_new_v = np.zeros((num_robots,))
        t_sample_hold = sample_hold_dt.sample()

        # Loop over time steps
        for t in range(int(env.max_episode_length)):
            # Decide on rom action
            new_v_nom = rom.sample_uniform_bounded_v(z[t, :, :])
            new_t_sample_hold = sample_hold_dt.sample()

            update_inds = (t_since_new_v >= t_sample_hold)

            v_nom[update_inds, :] = new_v_nom[update_inds, :]
            t_since_new_v[update_inds] = 0
            t_sample_hold[update_inds] = new_t_sample_hold[update_inds]
            t_since_new_v += 1

            vt = rom.clip_v_z(z[t, :, :], v_nom)

            # Execute rom action
            zt_p1 = rom.f(z[t, :, :], vt)

            # Get desired pose and velocity
            des_pose, des_vel = rom.des_pose_vel(z[t, :, :], vt)

            # Get robot pose
            robot_pose = np.hstack((base[:, :2], quat2yaw(base[:, 3:7])[:, None]))

            # Compute pose error
            err = des_pose - robot_pose
            err[:, :2] = np.squeeze(yaw2rot(robot_pose[:, 2]) @ err[:, :2][:, :, None])  # Place error in local frame
            if not track_yaw:
                err[:, 2] = 0

            # Compute control action via velocity tracking (P control)
            ut = des_vel + (Kp @ err.T).T
            ut = np.clip(ut, u_min, u_max)

            # Step environment
            obs[:, CMD] = torch.from_numpy(ut).float().to(env.device)
            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())

            # Save Data
            base = env.root_states.cpu().numpy()
            done[t, :] = dones.cpu().numpy()  # Termination should not be used for tube training
            u[t, :, :] = ut
            v[t, :, :] = vt
            x[t + 1, :, :] = get_state(base, env.dof_pos.cpu().numpy(), env.dof_vel.cpu().numpy())
            z[t + 1, :, :] = zt_p1
            z[t + 1, done[t, :], :] = rom.proj_z(base)[done[t, :], :]  # Terminated envs reset to zero tracking error
            pz_x[t + 1, :, :] = rom.proj_z(base)

        # Log Data
        # TODO: Write to local
        with open(f"{data_path}/epoch_{epoch}.pickle", "wb") as f:
            epoch_data = {
                'x': x,
                'u': u,
                'z': z,
                'v': v,
                'pz_x': pz_x,
                'done': done
            }
            pickle.dump(epoch_data, f)

    # TODO: Write to wandb
    artifact = wandb.Artifact(
        type="rom_tracking_data",
        name=f"{wandb.run.id}_rom_tracking_data"
    )
    artifact.add_dir(str(data_path))
    wandb.run.log_artifact(artifact)
    print("[INFO] Finished saving particles. Waiting for wandb to finish...")


if __name__ == "__main__":
    data_creation_main()
