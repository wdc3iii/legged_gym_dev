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

    # Generate grid for a single robot over the search space of -1 to 1 for both x and y with density 10 and deviation 0.025
    start = -1
    end = 1
    density = 10
    deviation = 0.025
    robot_grids = generate_robot_grids(start, end, density, num_robots, deviation)
    robot_grids_with_z = add_zero_z_coordinate(robot_grids)
    robot_grid_iterator = 0

    start_point = np.array([0.0, 0.0, 1.0])
    # direction_vector = np.array(robot_grids_with_z[robot_index][0]) # [0.5, 0.5, 0.0])  # Example direction vector
    direction_vector = np.array([-.5, .5, 0])
    desired_speed = np.linalg.norm(direction_vector)
    unnormalized_direction_vector = direction_vector.copy()
    direction_vector = direction_vector / desired_speed  # Normalize the direction vector
    current_position = start_point.copy()
    current_yaw = 0  # Convert direction vector to yaw

    # PD controller gains
    Kp = 0.8  # Proportional gain for position and yaw
    Kd = 0.0  # Derivative gain for position and yaw

    # Initialize previous error terms for derivative calculation
    prev_lateral_error = 0.0
    prev_longitudinal_error = 0.0
    prev_yaw_error = 0.0

    delta_t = env.dt  # Time step

    # Index for commands based on the given structure of obs
    CMD_START_IDX = 9  # start of commands in obs_buf
    CMD_LIN_VEL_X_IDX = CMD_START_IDX  # linear velocity x command index
    CMD_LIN_VEL_Y_IDX = CMD_START_IDX + 1  # linear velocity y command index
    CMD_ANG_VEL_YAW_IDX = CMD_START_IDX + 2  # angular velocity yaw command index

    positions = []
    velocities = []

    for i in range((density ** 2) * int(env.max_episode_length)):
        ideal_position = start_point + unnormalized_direction_vector * env.dt * ((i + 1) % int(env.max_episode_length))
        # Retrieve current velocities and update position and yaw
        base_lin_vel_x = env.base_lin_vel[robot_index, 0].item()
        base_lin_vel_y = env.base_lin_vel[robot_index, 1].item()
        base_ang_vel_yaw = env.base_ang_vel[robot_index, 2].item()

        # Update yaw and position
        current_yaw += base_ang_vel_yaw * delta_t
        current_yaw = (current_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize yaw
        current_position[0] += delta_t * (np.cos(current_yaw) * base_lin_vel_x - np.sin(current_yaw) * base_lin_vel_y)
        current_position[1] += delta_t * (np.sin(current_yaw) * base_lin_vel_x + np.cos(current_yaw) * base_lin_vel_y)

        # Calculate longitudinal error
        longitudinal_error = np.dot(current_position[:2] - start_point[:2], direction_vector[:2])

        # Calculate lateral error (perpendicular distance to the direction vector)
        lateral_error = \
        np.cross(np.append(current_position[:2] - start_point[:2], 0), np.append(direction_vector[:2], 0))[2]

        # Calculate yaw error
        desired_yaw = np.arctan2(direction_vector[1], direction_vector[0])
        yaw_error = desired_yaw - current_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # Compute control commands using PD control
        control_command_y = Kp * lateral_error + Kd * (lateral_error - prev_lateral_error) / delta_t
        control_command_yaw = Kp * yaw_error + Kd * (yaw_error - prev_yaw_error) / delta_t

        # Adjust control command for x to maintain desired speed
        control_command_x = np.sqrt(max(0, desired_speed ** 2 - control_command_y ** 2))

        # Save current errors for next derivative calculation
        prev_lateral_error = lateral_error
        prev_longitudinal_error = longitudinal_error
        prev_yaw_error = yaw_error

        # Save current position and velocity commands
        positions.append(current_position.copy())
        velocities.append([control_command_x, control_command_y, control_command_yaw])

        # Update observations with new commands
        obs[robot_index, CMD_LIN_VEL_X_IDX] = control_command_y * direction_vector[0]  # reversed this intentionally
        obs[robot_index, CMD_LIN_VEL_Y_IDX] = control_command_x * direction_vector[1]
        obs[robot_index, CMD_ANG_VEL_YAW_IDX] = control_command_yaw

        print(i)
        print(f"x: {obs[robot_index, CMD_LIN_VEL_X_IDX]}")
        print(f"y: {obs[robot_index, CMD_LIN_VEL_Y_IDX]}")
        print(f"yaw: {obs[robot_index, CMD_ANG_VEL_YAW_IDX]}")
        print('--------')
        print(f'longitudinal error: {longitudinal_error}')
        print(f'lateral error: {lateral_error}')
        print(f'yaw error: {yaw_error}')

        # Normal action inferences
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # Reset the PD controller state and save data at multiples of env.max_episode_length
        if (i + 1) % env.max_episode_length == 0:
            # Save the data to a CSV file
            filename = f'trajectory_data_{(i + 1) // env.max_episode_length}.csv'
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['time', 'position_x', 'position_y', 'position_z', 'traj_x', 'traj_y', 'traj_z',
                              'velocity_x', 'velocity_y', 'velocity_yaw']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for t, (pos, vel) in enumerate(zip(positions, velocities)):
                    writer.writerow({
                        'time': t * delta_t,
                        'position_x': pos[0],  # x_x
                        'position_y': pos[1],  # x_y
                        'position_z': pos[2],
                        'traj_x': ideal_position[0],
                        'traj_y': ideal_position[1],
                        'traj_z': ideal_position[2],
                        'velocity_x': vel[0],
                        'velocity_y': vel[1],
                        'velocity_yaw': vel[2]
                    })

            # Reset positions and velocities for the next episode
            positions = []
            velocities = []

            # Reset PD controller state
            current_position = start_point.copy()
            current_yaw = 0
            prev_lateral_error = 0.0
            prev_longitudinal_error = 0.0
            prev_yaw_error = 0.0

            # unnormalized_direction_vector = np.array(robot_grids_with_z[robot_index][robot_grid_iterator])
            # desired_speed = np.linalg.norm(unnormalized_direction_vector)
            # direction_vector = unnormalized_direction_vector / desired_speed if desired_speed != 0 else np.zeros_like(unnormalized_direction_vector)
            # robot_grid_iterator += 1
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
