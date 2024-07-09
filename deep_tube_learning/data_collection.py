import os
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import hydra
import torch
import numpy as np
from pathlib import Path
from hydra.utils import instantiate


from deep_tube_learning.utils import quat2yaw


CMD_START_IDX = 9
CMD_LIN_VEL_PAR_IDX = CMD_START_IDX
CMD_LIN_VEL_PERP_IDX = CMD_START_IDX + 1
CMD_ANG_VEL_YAW_IDX = CMD_START_IDX + 2
CMD = np.array([CMD_LIN_VEL_PAR_IDX, CMD_LIN_VEL_PERP_IDX, CMD_ANG_VEL_YAW_IDX])

def get_state(base, joints):
    return np.concatenate((base[:, :7], joints[:, :12], base[:, 7:], joints[:, 12:]), axis=1)

@hydra.main(
    config_path=str(Path(__file__).parent / "configs" / "data_generation"),
    config_name="default",
    version_base="1.2",
)
def data_creation_main(cfg):
    # Load configuration
    num_robots = cfg.num_robots
    rom = instantiate(cfg.reduced_order_model)
    sample_hold_dt = instantiate(cfg.sample_hold_dt)
    Kp = cfg.Kp
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

    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cfg.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Loop over epochs
    for epoch in range(cfg.epochs):
        # Data structures
        x = np.zeros((int(env.max_episode_length) + 1, num_robots, env.dof_state.T.shape[1] + env.root_states.shape[1]))  # Epochs, steps, states
        u = np.zeros((int(env.max_episode_length), num_robots, 3))
        z = np.zeros((int(env.max_episode_length) + 1, num_robots, rom.n))
        v = np.zeros((int(env.max_episode_length), num_robots, rom.m))
        dones = np.ones((num_robots,), dtype=bool)

        # Initialization
        x[0, :, :] = get_state(env.root_states.cpu().numpy(), env.dof_state.T.cpu().numpy())

        v_nom = rom.sample_uniform_bounded_v(z[0, :, :])
        t_since_new_v = torch.zeros((num_robots,))
        t_sample_hold = sample_hold_dt.sample()

        # Loop over timesteps
        for t in range(env.max_episode_length):
            # Any environments which have terminated should be reset to zero tracking error
            z[t, dones, :] = rom.proj_z(env.root_states).cpu().numpy()[dones, :]

            # Decide on rom action
            new_v_nom = rom.sample_uniform_bounded_v(z[t, :, :])
            new_t_sample_hold = sample_hold_dt.sample()

            update_inds = (t_since_new_v >= t_sample_hold).copy()

            v_nom[update_inds, :] = new_v_nom
            t_since_new_v[update_inds] = 0
            t_sample_hold[update_inds] = new_t_sample_hold
            t_since_new_v += 1

            vt = rom.clip_v_z(v_nom)

            # Execute rom action
            zt_p1 = rom.f(z[t, :, :], vt)

            # Get desired pose and velocity
            des_pose, des_vel = rom.des_pose_vel(z[t, :, :], vt)

            # Get robot pose
            robot_pose = np.vstack((env.root_states[:, :2], quat2yaw(env.root_states[:, 3:7])))

            # Compute pose error
            err = des_pose - robot_pose
            # TODO: Rotate properly
            err[:, :2] = rot(robot_pose[:, 2]) @ err[:, :2]  # Place error in local frame

            # Compute control action via velocity tracking (P control)
            ut = des_vel + Kp @ err
            ut = torch.clip(ut, u_min, u_max)

            # Step environment
            obs[:, CMD] = ut
            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())

            # Save Data
            u[t, :, :] = ut.cpu().numpy()
            v[t, :, :] = vt.cpu().numpy()
            x[t + 1, :, :] = get_state(env.root_states.cpu().numpy(), env.dof_state.T.cpu().numpy())
            z[t + 1, :, :] = zt_p1.cpu().numpy()

        # Log Data
        # TODO: Write to local
        # TODO: Write to wandb





if __name__ == "__main__":
    data_creation_main()
