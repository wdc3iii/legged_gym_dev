import os
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from rom_dynamics import SingleInt2D, DoubleInt2D, Unicycle, LateralUnicycle, ExtendedUnicycle, ExtendedLateralUnicycle
import numpy as np
import csv
import torch

data_prefix = "debugging_"
def initialize_rom(rom_type, dt):
    acc_max = 2
    alpha_max = 4
    vel_max = 0.5
    omega_max = 1
    pos_max = np.inf
    
    if rom_type == 'SingleInt2D':
        z_max = np.array([pos_max, pos_max])
        v_max = np.array([vel_max, vel_max])
        pm = SingleInt2D(dt, -z_max, z_max, -v_max, v_max, backend="numpy")
        initial_state = np.zeros(pm.n)
    elif rom_type == 'DoubleInt2D':
        z_max = np.array([pos_max, pos_max, vel_max, vel_max])
        v_max = np.array([acc_max, acc_max])
        pm = DoubleInt2D(dt, -z_max, z_max, -v_max, v_max, backend="numpy")
        initial_state = np.zeros(pm.n)
    elif rom_type == 'Unicycle':
        z_max = np.array([pos_max, pos_max, np.inf])
        v_max = np.array([vel_max, omega_max])
        v_min = -np.array([vel_max / 2, omega_max])
        pm = Unicycle(dt, -z_max, z_max, v_min, v_max, backend="numpy")
        initial_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
    elif rom_type == 'LateralUnicycle':
        z_max = np.array([pos_max, pos_max, np.inf])
        v_max = np.array([vel_max, vel_max / 2, omega_max])
        v_min = np.array([-vel_max / 2, -vel_max / 2, -omega_max])
        pm = LateralUnicycle(dt, -z_max, z_max, v_min, v_max, backend="numpy")
        initial_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
    elif rom_type == 'ExtendedUnicycle':
        z_max = np.array([pos_max, pos_max, np.inf, vel_max, omega_max])
        z_min = -np.array([pos_max, pos_max, np.inf, vel_max / 2, omega_max])
        v_max = np.array([acc_max, alpha_max])
        pm = ExtendedUnicycle(dt, z_min, z_max, -v_max, v_max, backend="numpy")
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, theta, v, omega]
    elif rom_type == 'ExtendedLateralUnicycle':
        z_max = np.array([pos_max, pos_max, np.inf, vel_max, vel_max / 2, omega_max])
        z_min = -np.array([pos_max, pos_max, np.inf, vel_max / 2, vel_max / 2, omega_max])
        v_max = np.array([acc_max, acc_max / 2, alpha_max])
        pm = ExtendedLateralUnicycle(dt, z_min, z_max, -v_max, v_max, backend="numpy")
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, theta, v, v_perp, omega]
    else:
        raise ValueError("Unsupported ROM type")
    
    return pm, initial_state

    
# Function to generate random unit vectors
def random_unit_vectors(num):
    angles = np.random.uniform(0, 2 * np.pi, num)
    return np.column_stack((np.cos(angles), np.sin(angles), np.zeros(num)))

def randomize_turn_intervals(base_interval, range_offset, num):
    return base_interval + np.random.randint(-range_offset, range_offset + 1, num)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    num_robots = 1  # Adjust the number of robots as needed
    randomness_scale = 0.5  # Scaling factor for randomness

    # Override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, num_robots)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.env.episode_length_s = 20                   # duration of episode (in seconds)
    env_cfg.noise.add_noise = False                     # domain randomizations:
    env_cfg.domain_rand.randomize_friction = False      # |
    env_cfg.domain_rand.friction_range = [.5, 1.25]     # |
    env_cfg.domain_rand.randomize_base_mass = False     # |
    env_cfg.domain_rand.added_mass_range = [-.5, .5]    # ---
    env_cfg.terrain.mesh_type = 'plane'                 # terrain randomization: switch to trimesh for random
    env_cfg.domain_rand.push_robots = False             # perturbation randomization:
    env_cfg.domain_rand.push_interval_s = 15            # | 
    env_cfg.domain_rand.max_push_vel_xy = 1.            # |

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
    num_iterations = 1000  # Number of iterations per episode
    Kp = 1.5
    Kd = 0.5

    def initialize_and_generate_trajectories(env):
        # Initialize SingleInt2D ROM
        dt = env.dt
        rom_type = 'SingleInt2D'
        sample_hold_min = 1
        sample_hold_max = 100
        pm, initial_state = initialize_rom(rom_type, dt)
        
        num_steps = num_iterations * int(env.max_episode_length)  # Number of steps in trajectory, adjust as needed
        start_positions = np.tile(initial_state, (num_robots, 1))

        def generate_trajectory(pm, start_position, num_steps):
            z0 = start_position
            zt = np.zeros((num_steps + 1, pm.n))
            vt = np.zeros((num_steps, pm.m))
            zt[0, :] = z0

            k = 0
            while k < num_steps:
                N = np.random.randint(sample_hold_min, sample_hold_max)
                v = pm.sample_uniform_bounded_v(zt[k, :])
                for k_step in range(N):
                    if k + k_step >= num_steps:
                        break
                    v_clip = pm.clip_v(zt[k + k_step, :], v)
                    vt[k + k_step, :] = v_clip
                    zt[k + k_step + 1, :] = pm.f(zt[k + k_step, :], v_clip)

                k += N

            return zt, vt

        trajectories = [generate_trajectory(pm, start_positions[robot_idx], num_steps) for robot_idx in range(num_robots)]
        return pm, start_positions, trajectories

    pm, start_positions, trajectories = initialize_and_generate_trajectories(env)

    # Initialize variables for PD control loop
    current_positions = np.zeros((num_robots, 3))
    perm_start_points = np.zeros((num_robots, 3))
    perm_start_points[:, :2] = start_positions.copy()
    current_yaws = np.zeros(num_robots)  # No yaw for SingleInt2D
    base_vels = np.array([1, 1])
    scale_down_from_vel = .85
    prev_position_errors = np.zeros((num_robots, 2))  # Only x and y
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
    desired_direction_vectors = np.zeros((num_robots, pm.m))
    max_x_speed = 2
    max_y_speed = 1
    max_yaw_rate = 2
    time_of_last_turns = np.zeros(num_robots)

    trajectories_positions = np.stack([traj[0] for traj in trajectories], axis=0)
    trajectories_directions = np.stack([traj[1] for traj in trajectories], axis=0)
    num_steps = num_iterations * int(env.max_episode_length)
    indices = np.minimum(np.arange(num_steps), trajectories_positions.shape[1] - 1)
    ideal_positions_steps = trajectories_positions[:, indices, :]
    desired_direction_vectors_steps = trajectories_directions[:, indices, :]

    for robot_idx in range(num_robots):
        ideal_positions[robot_idx].extend(ideal_positions_steps[robot_idx])
        unnormalized_direction_vectors_all[robot_idx].extend(desired_direction_vectors_steps[robot_idx])

    for i in range(num_steps):
        ideal_positions_step = ideal_positions_steps[:, i, :]
        desired_direction_vectors = desired_direction_vectors_steps[:, i, :]

        # TODO: Grab the heading yaw
        base_lin_vels = env.base_lin_vel.cpu().numpy()[:, :2]
        current_positions = env.root_states[:, :2].cpu().numpy()

        position_errors = ideal_positions_step - current_positions[:, :2]
        yaw_errors = np.zeros(num_robots)  # No yaw for SingleInt2D

        # TODO: Correct the yaw here
        control_commands_x = desired_direction_vectors[:, 0] + Kp * position_errors[:, 0] + Kd * (position_errors[:, 0] - prev_position_errors[:, 0]) / delta_t
        control_commands_y = desired_direction_vectors[:, 1] + Kp * position_errors[:, 1] + Kd * (position_errors[:, 1] - prev_position_errors[:, 1]) / delta_t
        control_commands_yaw = np.zeros(num_robots)  # No yaw for SingleInt2D

        # Scale commands
        # control_speeds = np.sqrt(control_commands_x**2 + control_commands_y**2)
        # scale = np.minimum(1, max_speed / control_speeds)
        # control_commands_x *= scale
        # control_commands_y *= scale

        # Clip commands
        control_commands_x = np.clip(control_commands_x, -max_x_speed, max_x_speed)
        control_commands_y = np.clip(control_commands_y, -max_y_speed, max_y_speed)
        control_commands_yaw = np.clip(control_commands_y, -max_yaw_rate, max_yaw_rate)

        prev_position_errors = position_errors
        prev_yaw_errors = yaw_errors

        for robot_idx in range(num_robots):
            temp_pos = current_positions[robot_idx].tolist()
            positions[robot_idx].append(temp_pos)
            velocities[robot_idx].append([control_commands_x[robot_idx], control_commands_y[robot_idx], 0])  # No yaw for SingleInt2D
            joint_positions[robot_idx].append(env.dof_pos[robot_idx].cpu().numpy().tolist())
            joint_velocities[robot_idx].append(env.dof_vel[robot_idx].cpu().numpy().tolist())

        for robot_idx in range(num_robots):
            obs[robot_idx, CMD_LIN_VEL_X_IDX] = control_commands_x[robot_idx]
            obs[robot_idx, CMD_LIN_VEL_Y_IDX] = control_commands_y[robot_idx]
            obs[robot_idx, CMD_ANG_VEL_YAW_IDX] = 0  # No yaw for SingleInt2D

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if (i + 1) % env.max_episode_length == 0:
            filename = f'rom_tracking_data/{data_prefix}trajectory_data_{(i + 1) // env.max_episode_length}.csv'
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
                            'position_yaw': 0,  # No yaw for SingleInt2D
                            'traj_x': ideal_pos[0],
                            'traj_y': ideal_pos[1],
                            'traj_yaw': 0,  # No yaw for SingleInt2D
                            'reduced_command_x': red[0],
                            'reduced_command_y': red[1],
                            'reduced_command_yaw': 0,  # No yaw for SingleInt2D
                            'velocity_x': vel[0],
                            'velocity_y': vel[1],
                            'velocity_yaw': 0,  # No yaw for SingleInt2D
                            'joint_positions': joint_positions[robot_idx][t],
                            'joint_velocities': joint_velocities[robot_idx][t]
                        })
            
            positions = [[] for _ in range(num_robots)]
            velocities = [[] for _ in range(num_robots)]
            ideal_positions = [[] for _ in range(num_robots)]
            unnormalized_direction_vectors_all = [[] for _ in range(num_robots)]

            # Reinitialize ROM and regenerate trajectories for the next episode
            pm, start_positions, trajectories = initialize_and_generate_trajectories(env)
            current_positions = start_positions.copy()
            perm_start_points = start_positions.copy()
            prev_position_errors = np.zeros((num_robots, 2))  # Only x and y
            prev_yaw_errors = np.zeros(num_robots)
            time_of_last_turns = np.zeros(num_robots)

            unnormalized_direction_vectors = random_unit_vectors(num_robots) * base_vels[0] * scale_down_from_vel * randomness_scale
            desired_direction_vectors = random_unit_vectors(num_robots)[:, :2]
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
