from legged_gym.envs.base.legged_robot_trajectory_config import LeggedRobotTrajectoryCfg, LeggedRobotTrajectoryCfgPPO

class HopperRoughTrajectoryCfg( LeggedRobotTrajectoryCfg ):
    class env( LeggedRobotTrajectoryCfg.env):
        num_envs = 4096 * 4
        num_observations = 38
        num_actions = 4  # Changes based on control type

    class terrain( LeggedRobotTrajectoryCfg.terrain):
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        mesh_type = 'plane'
        measure_heights = False

    class init_state(LeggedRobotTrajectoryCfg.init_state):
        pos = [0.0, 0.0, .3]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'foot_slide': 0.0,
            'wheel1_rotation': 0.0,
            'wheel2_rotation': 0.0,
            'wheel3_rotation': 0.0
        }
        randomize_yaw = True
        default_dof_pos_noise_lower = [-0.02, 0, 0, 0]  # [foot, wheel1, 2, 3]
        default_dof_pos_noise_upper = [0.02, 0, 0, 0]
        default_dof_vel_noise_lower = [-0.1, -100., -100., -100.]
        default_dof_vel_noise_upper = [0.1, 100., 100., 100.]
        default_root_pos_noise_lower = [-0.0, -0.0, -0.05, -0.03, -0.03, -0.03, -0.03]  # [x, y, z, qx, qy, qz, qw]
        default_root_pos_noise_upper = [0.0, 0.0, 0.05, 0.03, 0.03, 0.03, 0.03]
        default_root_vel_noise_lower = [-0.05, -0.05, -0.05, -0.2, -0.2, -0.2]  # [vx, vy, vz, wx, wy, wz]
        default_root_vel_noise_upper = [0.05, 0.05, 0.05, 0.2, 0.2, 0.2]

    class control(LeggedRobotTrajectoryCfg.control):
        # PD Drive parameters:
        stiffness = {
            'foot_slide': 400,
            'wheel1_rotation': 15.0,
            'wheel2_rotation': 15.0,
            'wheel3_rotation': 15.0
        }  # [N*m/rad for revolute, N/m for prismatic]

        damping = {
            'foot_slide': 40,
            'wheel1_rotation': 3.0,
            'wheel2_rotation': 3.0,
            'wheel3_rotation': 3.0
        }  # [N*m*s/rad for revolute, N*s/m for prismatic]

        wheel_spindown = {
            'wheel1_rotation': 0.1,
            'wheel2_rotation': 0.1,
            'wheel3_rotation': 0.1
        }  # [N*m/rad for revolute, N/m for prismatic]

        foot_pos_des = 0.03

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        control_type = "orientation"
        # control_type = "orientation_spindown"

        zero_action = [1.0, 0, 0, 0]

    class asset( LeggedRobotTrajectoryCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hopper/urdf/hopper.urdf'
        name = "hopper_flat_trajectory"
        foot_name = 'foot'
        terminate_after_contacts_on = ['wheel', 'torso']
        flip_visual_attachments = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

        # Foot spring properties
        spring_stiffness = 11732
        spring_damping = 50

        # TODO: actuator frame correct on hardware?
        rot_actuator = [
            [-0.8165, 0.2511, 0.2511],
            [-0, -0.7643, 0.7643],
            [-0.5773, -0.5939, -0.5939]
        ]
        wheel_speed_bounds = {
            'wheel1_rotation': 600,
            'wheel2_rotation': 600,
            'wheel3_rotation': 600
        }
        torque_speed_bound_ratio = 6
        disable_gravity = False

    class normalization(LeggedRobotTrajectoryCfg.normalization):
        class obs_scales:  # Note the foot states are not included in the observation, as foot sim is inaccurate
            lin_vel = 0.5
            ang_vel = 0.25
            dof_vel = 0.01
            z_pos = 1.0
            trajectory = [1.0, 1.0]
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotTrajectoryCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values
        class noise_scales:
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            z_pos = 0.02
            quat = 0.05
            height_measurements = 0.1

    class domain_rand(LeggedRobotTrajectoryCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        randomize_inv_base_mass = False
        inv_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_rom_distance = False
        max_rom_dist = None

        class rigid_shape_properties:
            randomize_restitution = False
            restitution_range = [0.0, 1.0]
            randomize_compliance = False
            compliance_range = [0.0, 1.0]
            randomize_thickness = False
            thickness_range = [0.0, 0.05]

        class dof_properties:
            randomize_stiffness = False
            added_stiffness_range = [-5.0, 5.0]
            randomize_damping = False
            added_damping_range = [-.2, .2]

        class spring_properties:
            randomize_stiffness = True
            stiffness_range = [0.9, 1.1]  # multiplicative
            randomize_damping = True
            damping_range = [0.9, 1.1]  # multiplicative
            randomize_setpoint = True
            setpoint_range = [0.75, 1.25]  # multiplicative

        class pd_gain_properties:
            randomize_p_gain = True
            p_gain_range = [0.9, 1.1]  # multiplicative
            randomize_d_gain = True
            d_gain_range = [0.9, 1.1]  # multiplicative

        class torque_speed_properties:
            randomize_max_torque = True
            max_torque_range = [0.95, 1.05]  # multiplicative
            randomize_max_speed = True
            max_speed_range = [0.9, 1.1]  # multiplicative
            randomize_slope = True
            slope_range = [0.9, 1.1]  # multiplicative

    class rewards(LeggedRobotTrajectoryCfg.rewards):
        class scales(LeggedRobotTrajectoryCfg.rewards.scales):
            # orientation = -1.
            collision = -1.
            # action_rate = -0.1
            # torques = -0.00001
            # dof_acc = -2.5e-7
        #     differential_error = 2.

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = .55
        max_contact_force = 100.  # forces above this value are penalized

        class raibert:
            Kp = -0.3
            Kv = -0.9
            Kff = 0.0
            clip_pos = 0.5
            clip_vel = 1.0
            clip_ang = 0.2

        class differential_error:
            pos_slope = 4
            neg_slope = 1
        class sigma_values:
            tracking_rom = 0.25  # tracking reward = exp(-error^2/sigma)
            feet_air_time = 1.
            stumble = 1.
            stand_still = 1.
            feet_contact_forces = 1.
            tracking_lin_vel = 1.
            tracking_ang_vel = 1.
            torque_limits = 1.
            dof_vel_limits = 1.
            dof_pos_limits = 1.
            termination = 1.
            collision = 1.
            action_rate = 1.
            dof_acc = 1.
            dof_vel = 1.
            torques = 1.
            base_height = 1.
            orientation = 1.
            ang_vel_xy = 1.
            lin_vel_z = 1.

    class curriculum:
        use_curriculum = False
        curriculum_steps = [2500, 5000]

        class push:
            magnitude = [0.1, 0.5, 1]  # multiplier
            time = [3, 2, 1]  # multiplier

        class trajectory_generator:
            weight_sampler = ['WeightSamplerSampleAndHold', 'WeightSamplerSampleAndHold', 'WeightSamplerSampleAndHold']
            t_low = [3, 2, 1]  # multiplier
            t_high = [3, 2, 1]  # multiplier
            freq_low = [0.01, 0.1, 1]  # multiplier
            freq_high = [0.1, 0.5, 1]  # multiplier

        class rom:
            z = [1, 1, 1]
            v = [0.5, 0.75, 1]

        class sigma:
            tracking_rom = [1., .8, .6]

        class rewards:
            tracking_rom = [1., .8, .6]
            feet_air_time = [1., .8, .6]
            stumble = [1., .8, .6]
            stand_still = [1., .8, .6]
            feet_contact_forces = [1., .8, .6]
            tracking_lin_vel = [1., .8, .6]
            tracking_ang_vel = [1., .8, .6]
            torque_limits = [1., .8, .6]
            dof_vel_limits = [1., .8, .6]
            dof_pos_limits = [1., .8, .6]
            termination = [1., .8, .6]
            collision = [1., .8, .6]
            action_rate = [1., .8, .6]
            dof_acc = [1., .8, .6]
            dof_vel = [1., .8, .6]
            torques = [1., .8, .6]
            base_height = [1., .8, .6]
            orientation = [1., .8, .6]
            ang_vel_xy = [1., .8, .6]
            lin_vel_z = [1., .8, .6]
            differential_error = [1., .8, .6]
            raibert = [1., 1., 1.]
    class policy_model:
        policy_to_use = 'rh'
        class rh:
            K_p = -.3
            K_v = -.9
            clip_value_pos = .1
            clip_value_vel = 1.
            clip_value_total = 1.

class HopperRoughTrajectoryCfgPPO( LeggedRobotTrajectoryCfgPPO ):
    class policy(LeggedRobotTrajectoryCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(LeggedRobotTrajectoryCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotTrajectoryCfgPPO.runner):
        run_name = ''
        experiment_name = 'hopper_flat_trajectory'
        load_run = -1
        max_iterations = 1500
