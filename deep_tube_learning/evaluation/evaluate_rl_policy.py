import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import os
import hydra
import wandb
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from deep_tube_learning.utils import (update_args_from_hydra, update_cfgs_from_hydra, quat2yaw, yaw2rot, wrap_angles,
                                      wandb_model_load, update_hydra_cfg)
from trajopt.rom_dynamics import ZeroTrajectoryGenerator, CircleTrajectoryGenerator, SquareTrajectoryGenerator


def get_state(base, joint_pos, joint_vel):
    return torch.concatenate((base[:, :7], joint_pos, base[:, 7:], joint_vel), dim=1)

def evaluate(traj_cls, push_robots, curriculum_state=3):
    steps = 1000
    exp_name = "coleonguard-Georgia Institute of Technology/RL_Training/5y4goa2v"
    model_name = f'{exp_name}_model:best{curriculum_state}'
    api = wandb.Api()
    rl_cfg, state_dict = wandb_model_load(api, model_name)

    # Overwrite some params
    rl_cfg.headless = False
    rl_cfg.args.headless = False
    rl_cfg.env_config.env.num_envs = 10
    rl_cfg.env_config.env.episode_length_s = 20
    rl_cfg.env_config.trajectory_generator.cls = traj_cls
    rl_cfg.env_config.trajectory_generator.t_low = 21
    rl_cfg.env_config.trajectory_generator.t_high = 21
    rl_cfg.env_config.domain_rand.randomize_friction = False
    rl_cfg.env_config.domain_rand.randomize_base_mass = False
    rl_cfg.env_config.domain_rand.randomize_inv_base_mass = False
    rl_cfg.env_config.domain_rand.push_robots = push_robots
    rl_cfg.env_config.domain_rand.randomize_base_mass = False
    rl_cfg.env_config.domain_rand.randomize_base_mass = False
    rl_cfg.env_config.domain_rand.randomize_base_mass = False
    rl_cfg.env_config.domain_rand.rigid_shape_properties.randomize_restitution = False
    rl_cfg.env_config.domain_rand.rigid_shape_properties.randomize_compliance = False
    rl_cfg.env_config.domain_rand.rigid_shape_properties.randomize_thickness = False
    rl_cfg.env_config.domain_rand.dof_properties.randomize_stiffness = False
    rl_cfg.env_config.domain_rand.dof_properties.randomize_damping = False
    rl_cfg.env_config.domain_rand.spring_properties.randomize_stiffness = False
    rl_cfg.env_config.domain_rand.spring_properties.randomize_damping = False
    rl_cfg.env_config.domain_rand.spring_properties.randomize_setpoint = False
    rl_cfg.env_config.domain_rand.pd_gain_properties.randomize_p_gain = False
    rl_cfg.env_config.domain_rand.pd_gain_properties.randomize_d_gain = False
    rl_cfg.env_config.domain_rand.torque_speed_properties.randomize_max_torque = False
    rl_cfg.env_config.domain_rand.torque_speed_properties.randomize_max_speed = False
    rl_cfg.env_config.domain_rand.torque_speed_properties.randomize_slope = False
    rl_cfg.env_config.curriculum.use_curriculum = False

    args = get_args()
    args = update_args_from_hydra(rl_cfg, args)
    env_cfg, train_cfg = task_registry.get_cfgs(rl_cfg.task)
    env_cfg, train_cfg = update_cfgs_from_hydra(rl_cfg, env_cfg, train_cfg)

    env, env_cfg = task_registry.make_env(name=rl_cfg.task, args=args, env_cfg=env_cfg)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # load in the model
    ppo_runner.alg.actor_critic.load_state_dict(state_dict['model_state_dict'])
    ppo_runner.alg.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    x_n = env.dof_pos.shape[1] + env.dof_vel.shape[1] + env.root_states.shape[1]

    # Loop over steps
    num_robots = env_cfg.env.num_envs
    rom_n = env.rom.n
    rom_m = env.rom.m

    x = torch.zeros((steps + 1, num_robots, x_n), device='cuda')  # Epochs, steps, states
    z = torch.zeros((steps + 1, num_robots, rom_n), device='cuda')
    pz_x = torch.zeros((steps + 1, num_robots, rom_n), device='cuda')

    # Initialization
    base = env.root_states
    x[0, :, :] = get_state(base, env.dof_pos, env.dof_vel)
    z[0, :, :] = env.rom.proj_z(base)
    pz_x[0, :, :] = env.rom.proj_z(base)

    env.traj_gen.reset(env.rom.proj_z(env.root_states))
    for t in range(steps):
        # Step environment
        actions = policy(obs.detach())
        obs, _, _, done, _ = env.step(actions.detach())

        # Save Data
        base = env.root_states
        proj = env.rom.proj_z(base)
        x[t + 1, :, :] = get_state(base, env.dof_pos, env.dof_vel)
        z[t + 1, :, :] = env.traj_gen.trajectory[:, 0, :]
        z[t + 1, done, :] = proj[done, :]  # Terminated envs reset to zero tracking error
        pz_x[t + 1, :, :] = env.rom.proj_z(base)

    z = z.cpu().numpy()
    pz_x = pz_x.cpu().numpy()

    # Plot the trajectories after the loop
    fig, ax = plt.subplots()
    env.rom.plot_spacial(ax, z[:, 0, :], '.-k')
    env.rom.plot_spacial(ax, pz_x[:, 0, :], '.-b')
    plt.show()
    fig, ax = plt.subplots()
    env.rom.plot_spacial(ax, pz_x[:, 1, :], '.-b')
    env.rom.plot_spacial(ax, z[:, 1, :], '.-k')
    plt.show()


if __name__ == "__main__":
    # evaluate('ZeroTrajectoryGenerator', True)
    evaluate('SquareTrajectoryGenerator', False)
    # evaluate('CircleTrajectoryGenerator', False)

