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
    num_robots = 1000  # Adjust the number of robots as needed
    randomness_scale = 0.5  # Scaling factor for randomness

    # Override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, num_robots)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # Prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Export policy as a jit module (used to run it from C++)
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
    start_points = np.array([[0.0, 0.0, 1.0]] * num_robots)
    max_speed = 2  # hard maximum speed cap
    num_iterations = 1000

    # Function to generate random unit vectors
    def random_unit_vectors(num):
        angles = np.random.uniform(0, 2 * np.pi, num)
        return np.column_stack((np.cos(angles), np.sin(angles), np.zeros(num)))

    def randomize_turn_intervals(base_interval, range_offset, num):
        return base_interval + np.random.randint(-range_offset, range_offset + 1, num)

    current_positions = start_points.copy()
    perm_start_points = start_points.copy()
    current_yaws = np.zeros(num_robots)
    base_vels = np.array([1, 1])
    scale_down_from_vel = .95 # planning model can move only .95 times as fast as normal robot
    base_turn_interval = 500  # number of iterations between each turn of the robot
    turn_interval_range = 150
    turn_intervals = randomize_turn_intervals(base_turn_interval, turn_interval_range, num_robots)

    # Randomly generate initial direction and desired direction vectors
    unnormalized_direction_vectors = random_unit_vectors(num_robots) * base_vels[0] * scale_down_from_vel * randomness_scale
    desired_direction_vectors = random_unit_vectors(num_robots)[:, :2]  # 2D vector for yaw calculation

    # PD controller gains
    Kp = 1.5
    Kd = 0.5

    # Initialize previous error terms
    prev_position_errors = np.zeros((num_robots, 2))
    prev_yaw_errors = np.zeros(num_robots)

    delta_t = env.dt
    CMD_START_IDX = 9
    CMD_LIN_VEL_X_IDX = CMD_START_IDX
    CMD_LIN_VEL_Y_IDX = CMD_START_IDX + 1
    CMD_ANG_VEL_YAW_IDX = CMD_START_IDX + 2

    positions = [[] for _ in range(num_robots)]
    velocities = [[] for _ in range(num_robots)]
    ideal_positions = [[] for _ in range(num_robots)]
    joint_positions = [[] for _ in range(num_robots)]
    joint_velocities = [[] for _ in range(num_robots)]
    unnormalized_direction_vectors_all = [[] for _ in range(num_robots)]
    time_of_last_turns = np.zeros(num_robots)

    for i in range(num_iterations * int(env.max_episode_length)):
        turn_mask = (i - time_of_last_turns) >= turn_intervals
        if np.any(turn_mask):
            unnormalized_direction_vectors[turn_mask] = random_unit_vectors(np.sum(turn_mask)) * base_vels[0] * scale_down_from_vel * randomness_scale
            desired_direction_vectors[turn_mask] = random_unit_vectors(np.sum(turn_mask))[:, :2]  # 2D vector for yaw calculation
            turn_intervals[turn_mask] = randomize_turn_intervals(base_turn_interval, turn_interval_range, np.sum(turn_mask))
            time_of_last_turns[turn_mask] = i
            start_points[turn_mask] = current_positions[turn_mask]
            print(f'Turn at iteration {i}, new directions for robots: {unnormalized_direction_vectors[turn_mask]}, new desired directions: {desired_direction_vectors[turn_mask]}, will turn again in {turn_intervals[turn_mask]} intervals')

        ideal_positions_step = start_points + unnormalized_direction_vectors * env.dt * (i - time_of_last_turns).reshape(-1, 1)
        for robot_idx in range(num_robots):
            ideal_positions[robot_idx].append(ideal_positions_step[robot_idx].copy())
            temp_dir_vecs = unnormalized_direction_vectors[robot_idx].tolist()
            temp_dir_vecs.append(desired_direction_vectors[robot_idx])
            unnormalized_direction_vectors_all[robot_idx].append(temp_dir_vecs)

        # Update current positions and yaws
        base_lin_vels = env.base_lin_vel.cpu().numpy()[:, :2]
        base_ang_vel_yaws = env.base_ang_vel[:, 2]

        current_yaws = np.add(current_yaws, base_ang_vel_yaws.cpu().numpy() * delta_t)
        current_yaws = (current_yaws + np.pi) % (2 * np.pi) - np.pi
        current_positions[:, 0] += delta_t * (np.cos(current_yaws) * base_lin_vels[:, 0] - np.sin(current_yaws) * base_lin_vels[:, 1])
        current_positions[:, 1] += delta_t * (np.sin(current_yaws) * base_lin_vels[:, 0] + np.cos(current_yaws) * base_lin_vels[:, 1])

        # Calculate position and yaw errors
        position_errors = ideal_positions_step[:, :2] - current_positions[:, :2]
        desired_yaws = np.arctan2(desired_direction_vectors[:, 1], desired_direction_vectors[:, 0])
        yaw_errors = desired_yaws - current_yaws
        yaw_errors = (yaw_errors + np.pi) % (2 * np.pi) - np.pi

        # PD control for position and yaw
        control_commands_x = unnormalized_direction_vectors[:, 0] + Kp * position_errors[:, 0] + Kd * (position_errors[:, 0] - prev_position_errors[:, 0]) / delta_t
        control_commands_y = unnormalized_direction_vectors[:, 1] + Kp * position_errors[:, 1] + Kd * (position_errors[:, 1] - prev_position_errors[:, 1]) / delta_t
        control_commands_yaw = Kp * yaw_errors + Kd * (yaw_errors - prev_yaw_errors) / delta_t

        # Cap the control commands to max_speed
        control_speeds = np.sqrt(control_commands_x**2 + control_commands_y**2)
        scale = np.minimum(1, max_speed / control_speeds)
        control_commands_x *= scale
        control_commands_y *= scale

        # Save current errors
        prev_position_errors = position_errors
        prev_yaw_errors = yaw_errors

        # Save positions and velocities
        for robot_idx in range(num_robots):
            temp_pos = current_positions[robot_idx].tolist()
            temp_pos.append(current_yaws[robot_idx])
            positions[robot_idx].append(temp_pos)
            velocities[robot_idx].append([control_commands_x[robot_idx], control_commands_y[robot_idx], control_commands_yaw[robot_idx]])

            # Assuming legged_robot is an instance of LeggedRobot class
            joint_positions[robot_idx].append(env.dof_pos[robot_idx].cpu().numpy().tolist())
            joint_velocities[robot_idx].append(env.dof_vel[robot_idx].cpu().numpy().tolist())

        # Update observations with new commands
        for robot_idx in range(num_robots):
            obs[robot_idx, CMD_LIN_VEL_X_IDX] = control_commands_x[robot_idx]
            obs[robot_idx, CMD_LIN_VEL_Y_IDX] = control_commands_y[robot_idx]
            obs[robot_idx, CMD_ANG_VEL_YAW_IDX] = 0 # control_commands_yaw[robot_idx]

        # Normal action inferences
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if (i + 1) % env.max_episode_length == 0:
            # Save the data to a CSV file
            filename = f'trajectory_data_{(i + 1) // env.max_episode_length}.csv'
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['time', 'episode_number', 'robot_index', 'position_x', 'position_y', 'position_yaw', 'traj_x', 'traj_y', 'traj_yaw', 'reduced_command_x',
                            'reduced_command_y', 'reduced_command_yaw', 'velocity_x', 'velocity_y', 'velocity_yaw', 'joint_positions', 'joint_velocities']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for t in range(len(positions[0])):
                    for robot_idx in range(num_robots):
                        pos = positions[robot_idx][t]
                        vel = velocities[robot_idx][t]
                        red = unnormalized_direction_vectors_all[robot_idx][t]
                        ideal_pos = ideal_positions[robot_idx][t]
                        writer.writerow({
                            'time': t * delta_t,
                            'episode_number': (i + 1) // env.max_episode_length,
                            'robot_index': robot_idx,
                            'position_x': pos[0],
                            'position_y': pos[1],
                            'position_yaw': pos[2],
                            'traj_x': ideal_pos[0],
                            'traj_y': ideal_pos[1],
                            'traj_yaw': red[2],
                            'reduced_command_x': red[0],
                            'reduced_command_y': red[1],
                            'reduced_command_yaw': red[2],
                            'velocity_x': vel[0],
                            'velocity_y': vel[1],
                            'velocity_yaw': vel[2],
                            'joint_positions': joint_positions[robot_idx][t],
                            'joint_velocities': joint_velocities[robot_idx][t]
                        })

            
            # Reset positions and velocities
            positions = [[] for _ in range(num_robots)]
            velocities = [[] for _ in range(num_robots)]
            ideal_positions = [[] for _ in range(num_robots)]
            unnormalized_direction_vectors_all = [[] for _ in range(num_robots)]

            # Reset PD controller state
            start_points = perm_start_points.copy()
            current_positions = start_points.copy()
            current_yaws = np.zeros(num_robots)
            prev_position_errors = np.zeros((num_robots, 2))
            prev_yaw_errors = np.zeros(num_robots)
            time_of_last_turns = np.zeros(num_robots)

            # Randomly generate new direction and desired direction vectors
            unnormalized_direction_vectors = random_unit_vectors(num_robots) * base_vels[0] * scale_down_from_vel * randomness_scale
            desired_direction_vectors = random_unit_vectors(num_robots)[:, :2]  # 2D vector for yaw calculation
            print(f'Changed to {unnormalized_direction_vectors}')

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
