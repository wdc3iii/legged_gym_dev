from legged_gym.utils import get_args, task_registry

import wandb
from isaacgym.torch_utils import *
import numpy as np
import matplotlib.pyplot as plt

from deep_tube_learning.utils import (update_args_from_hydra, update_cfgs_from_hydra, wandb_model_load)
from deep_tube_learning.raibert import RaibertHeuristic
from trajopt.rom_dynamics import SingleInt2D
def get_state(base, joint_pos, joint_vel):
    return torch.concatenate((base[:, :7], joint_pos, base[:, 7:], joint_vel), dim=1)

def evaluate(traj_cls, push_robots, curriculum_state=0):
    steps = 1000
    exp_name = "coleonguard-Georgia Institute of Technology/RL_Training/z3cpmz9n"
    model_name = f'{exp_name}_model:best{curriculum_state}'                         # even if using rayburn heuristic, load in a RL model for env settings
    api = wandb.Api()
    rl_cfg, state_dict = wandb_model_load(api, model_name)

    # Overwrite some params
    rl_cfg.headless = False
    rl_cfg.args.headless = False
    rl_cfg.env_config.env.num_envs = 1
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
    rl_cfg.env_config.domain_rand.randomize_rom_distance = False
    rl_cfg.env_config.domain_rand.torque_speed_properties.randomize_max_torque = False
    rl_cfg.env_config.domain_rand.torque_speed_properties.randomize_max_speed = False
    rl_cfg.env_config.domain_rand.torque_speed_properties.randomize_slope = False
    rl_cfg.env_config.curriculum.use_curriculum = False

    rl_cfg.env_config.policy_model.rh.K_p = 0.5
    rl_cfg.env_config.policy_model.rh.K_v = 1.5

    args = get_args()
    args = update_args_from_hydra(rl_cfg, args)
    env_cfg, train_cfg = task_registry.get_cfgs(rl_cfg.task)
    env_cfg, train_cfg = update_cfgs_from_hydra(rl_cfg, env_cfg, train_cfg)

    env, env_cfg = task_registry.make_env(name=rl_cfg.task, args=args, env_cfg=env_cfg)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # load in the model
    ppo_runner.alg.actor_critic.load_state_dict(state_dict['model_state_dict'])
    ppo_runner.alg.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    if rl_cfg.env_config.policy_model.policy_to_use == 'rl':
        policy = ppo_runner.get_inference_policy(device=env.device)
    elif rl_cfg.env_config.policy_model.policy_to_use == 'rh':
        raibert = RaibertHeuristic(rl_cfg)
        policy = raibert.get_inference_policy(device=env.device)

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
    actions_list = []
    quaternion_list = []
    wheel_torques = []
    # minus 1 because we're computing differences between position to get velocity looking forward
    for t in range(steps - 1):
        wheel_torques.append(env.dof_vel[0, 1:4].cpu().detach().numpy())
        # Step environment
        # have to modify obs if using Raibert Heuristic
        if rl_cfg.env_config.policy_model.policy_to_use == 'rh':
            if isinstance(env.traj_gen.rom, SingleInt2D):
                current_velocity = quat_rotate_inverse(env.root_states[:, 3:7], env.root_states[:, 7:10])[:, :2]
                current_position = env.root_states[:, :2]

                desired_position = env.traj_gen.trajectory[:, 0]
                desired_velocity = env.traj_gen.trajectory[:, 1, :] - env.traj_gen.trajectory[:, 0, :]

                positional_error = desired_position - current_position
                velocity_error = desired_velocity - current_velocity
                quaternion = env.root_states[:, 3:7]  # w,x,y,z
                obs = torch.cat((positional_error, velocity_error, quaternion), dim=1)
        actions = policy(obs.detach())
        obs, _, _, done, _ = env.step(actions.detach())

        actions_list.append(actions[0].cpu().numpy())
        quaternion_list.append(env.root_states[0, 3:7].cpu().numpy())

        # Save Data
        base = env.root_states
        proj = env.rom.proj_z(base)
        x[t + 1, :, :] = get_state(base, env.dof_pos, env.dof_vel)
        z[t + 1, :, :] = env.traj_gen.trajectory[:, 0, :]
        z[t + 1, done, :] = proj[done, :]  # Terminated envs reset to zero tracking error
        pz_x[t + 1, :, :] = env.rom.proj_z(base)

    z = z.cpu().numpy()
    pz_x = pz_x.cpu().numpy()

    wheel_torques = np.array(wheel_torques)
    time_steps = np.arange(wheel_torques.shape[0])

    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(time_steps, wheel_torques[:, i], label=f'Wheel {i + 1} Torque')

    plt.title('Wheel Torques Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Torque')
    plt.legend()
    plt.grid(True)
    plt.show()

    actions_array = np.array(actions_list)
    quaternion_array = np.array(quaternion_list)

    # Plot each pair of elements from actions and quaternions
    # for i in range(min(actions_array.shape[1], quaternion_array.shape[1])):
    plt.figure()
    plt.plot(quaternion_array)
    plt.gca().set_prop_cycle(None)
    plt.plot(actions_array[:, [1, 2, 3, 0]], linestyle='--')
    plt.title(f'Action and Quaternion Element {i} for the First Robot')
    plt.legend(['qx', 'qy', 'qz', 'qw', 'qdx', 'qdy', 'qdz', 'qdw'])
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.show()

    # Plot the trajectories after the loop
    fig, ax = plt.subplots()
    env.rom.plot_spacial(ax, pz_x[2:-2, 0, :], '.-b')
    env.rom.plot_spacial(ax, z[2:-2, 0, :], '.-k')

    plt.show()
    fig, ax = plt.subplots()
    env.rom.plot_spacial(ax, z[2:-2, 1, :], '.-k')
    env.rom.plot_spacial(ax, pz_x[2:-2, 1, :], '.-b')
    plt.show()


if __name__ == "__main__":
    # evaluate('ZeroTrajectoryGenerator', True)
    evaluate('SquareTrajectoryGenerator', False)
    # evaluate('CircleTrajectoryGenerator', False)

