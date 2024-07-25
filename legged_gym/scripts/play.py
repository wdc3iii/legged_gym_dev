import os
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from trajopt.rom_dynamics import SingleInt2D, DoubleInt2D, Unicycle, LateralUnicycle, ExtendedUnicycle, ExtendedLateralUnicycle
import numpy as np
import csv
import torch
from scipy.spatial.transform import Rotation

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # TODO: debugging isaacgym -> mujoco

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

        onnx_path = f"{path}/{type(env_cfg).__name__}.onnx"
        torch.onnx.export(
            ppo_runner.alg.actor_critic.actor,
            obs[0],
            onnx_path,
            export_params=True
        )
        print('Exported policy as onnx file to: ', onnx_path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    N = int(env.max_episode_length)
    command = np.zeros((N, len(env.commands[0])))
    action = np.zeros((N, 4))
    torque = np.zeros((N, 4))

    pos = np.zeros((N, 3))
    quat = np.zeros((N, 4))
    dof = np.zeros((N, 4))

    vel = np.zeros((N, 3))
    omega = np.zeros((N, 3))
    ddof = np.zeros((N, 4))

    for i in range(1 * int(env.max_episode_length)):
        command[i, :] = env.commands[0].cpu().numpy()
        pos[i, :] = env.root_states[0, :3].cpu().numpy()
        quat[i, :] = env.root_states[0, 3:7].cpu().numpy()
        dof[i, :] = env.dof_pos[0, :].cpu().numpy()

        vel[i, :] = env.root_states[0, 7:10].cpu().numpy()
        omega[i, :] = env.root_states[0, -3:].cpu().numpy()
        ddof[i, :] = env.dof_vel[0, :].cpu().numpy()

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        action[i, :] = actions[0].detach().cpu().numpy()
        torque[i, :] = env.torques[0, :].cpu().numpy()

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                        'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

    import scipy
    scipy.io.savemat('play_data.mat', {
        "cmd": command,
        "action": action,
        "pos": pos,
        "quat": quat,
        "dof": dof,
        "vel": vel,
        "omega": omega,
        "ddof": ddof,
        "torque": torque
    })


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)