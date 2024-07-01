from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import quaternion_to_direction_vector, random_vector, generate_grid_search_configs_2d, \
    generate_robot_grids, add_zero_z_coordinate
import numpy as np
import csv
import torch
import time


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    num_robots = 1  # affects grid space searching too
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, num_robots)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
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

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    
    # Initialize variables
    start_point = np.array([0.0, 0.0, 1.0])
    max_speed = 1.0  # hard maximum speed cap
    num_iterations = 5

    # Function to generate a random unit vector
    def random_unit_vector():
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle), 0.0])

    # Randomly generate initial direction and desired direction vectors
    unnormalized_direction_vector = random_unit_vector()
    desired_direction_vector = random_unit_vector()[:2]  # 2D vector for yaw calculation

    current_position = start_point.copy()
    current_yaw = 0
    base_vels = [.5,.5]

    # PD controller gains
    Kp = 0.8
    Kd = 0.0

    # Initialize previous error terms
    prev_position_error = np.array([0.0, 0.0])
    prev_yaw_error = 0.0

    delta_t = env.dt
    CMD_START_IDX = 9
    CMD_LIN_VEL_X_IDX = CMD_START_IDX
    CMD_LIN_VEL_Y_IDX = CMD_START_IDX + 1
    CMD_ANG_VEL_YAW_IDX = CMD_START_IDX + 2

    positions = []
    velocities = []

    for i in range(num_iterations * int(env.max_episode_length)):
        ideal_position = start_point + unnormalized_direction_vector * env.dt * ((i+1) % int(env.max_episode_length))
        
        # Update current position and yaw
        base_lin_vel_x = env.base_lin_vel[robot_index, 0].item()
        base_lin_vel_y = env.base_lin_vel[robot_index, 1].item()
        base_ang_vel_yaw = env.base_ang_vel[robot_index, 2].item()

        current_yaw += base_ang_vel_yaw * delta_t
        current_yaw = (current_yaw + np.pi) % (2 * np.pi) - np.pi
        current_position[0] += delta_t * (np.cos(current_yaw) * base_lin_vel_x - np.sin(current_yaw) * base_lin_vel_y)
        current_position[1] += delta_t * (np.sin(current_yaw) * base_lin_vel_x + np.cos(current_yaw) * base_lin_vel_y)

        # Calculate position error
        position_error = ideal_position[:2] - current_position[:2]

        # Calculate yaw error using desired_direction_vector
        desired_yaw = np.arctan2(desired_direction_vector[1], desired_direction_vector[0])
        yaw_error = desired_yaw - current_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

        # PD control for position and yaw
        control_command_x = base_vels[0] + Kp * position_error[0] + Kd * (position_error[0] - prev_position_error[0]) / delta_t
        control_command_y = base_vels[1] + Kp * position_error[1] + Kd * (position_error[1] - prev_position_error[1]) / delta_t
        control_command_yaw = Kp * yaw_error + Kd * (yaw_error - prev_yaw_error) / delta_t

        # Cap the control commands to max_speed
        control_speed = np.sqrt(control_command_x**2 + control_command_y**2)
        if control_speed > max_speed:
            scale = max_speed / control_speed
            control_command_x *= scale
            control_command_y *= scale

        # Save current errors
        prev_position_error = position_error
        prev_yaw_error = yaw_error

        # Save positions and velocities
        positions.append(current_position.copy())
        velocities.append([control_command_x, control_command_y, control_command_yaw])

        # Update observations with new commands
        obs[robot_index, CMD_LIN_VEL_X_IDX] = control_command_x
        obs[robot_index, CMD_LIN_VEL_Y_IDX] = control_command_y
        obs[robot_index, CMD_ANG_VEL_YAW_IDX] = control_command_yaw

        # Normal action inferences
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if (i + 1) % env.max_episode_length == 0:
            # Save the data to a CSV file
            filename = f'trajectory_data_{(i + 1) // env.max_episode_length}.csv'
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['time', 'position_x', 'position_y', 'position_z', 'traj_x', 'traj_y', 'traj_z', 'velocity_x', 'velocity_y', 'velocity_yaw']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for t, (pos, vel) in enumerate(zip(positions, velocities)):
                    writer.writerow({
                        'time': t * delta_t,
                        'position_x': pos[0],
                        'position_y': pos[1],
                        'position_z': pos[2],
                        'traj_x': ideal_position[0],
                        'traj_y': ideal_position[1],
                        'traj_z': ideal_position[2],
                        'velocity_x': vel[0],
                        'velocity_y': vel[1],
                        'velocity_yaw': vel[2]
                    })
            
            # Reset positions and velocities
            positions = []
            velocities = []

            # Reset PD controller state
            current_position = start_point.copy()
            current_yaw = 0
            prev_position_error = np.array([0.0, 0.0])
            prev_yaw_error = 0.0

            # Randomly generate new direction and desired direction vectors
            unnormalized_direction_vector = random_unit_vector()
            desired_direction_vector = random_unit_vector()[:2]  # 2D vector for yaw calculation
            
            print(f'Changed to {unnormalized_direction_vector}')

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


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
