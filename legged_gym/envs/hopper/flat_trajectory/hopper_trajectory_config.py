from legged_gym.envs.base.legged_robot_trajectory_config import LeggedRobotTrajectoryCfg, LeggedRobotTrajectoryCfgPPO

class HopperRoughTrajectoryCfg( LeggedRobotTrajectoryCfg ):
    class env( LeggedRobotTrajectoryCfg.env):
        num_envs = 4096 * 4
        num_observations = 40
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
        class obs_scales:
            lin_vel = 0.5
            ang_vel = 0.25
            foot_pos = 0.0
            foot_vel = 0.0
            dof_vel = 0.01
            z_pos = 1.0
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotTrajectoryCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values
        class noise_scales:
            foot_pos = 0.01
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
        randomize_inv_base_mass = True
        inv_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

        class rigid_shape_properties:
            randomize_restitution = True
            restitution_range = [0.0, 1.0]
            randomize_compliance = True
            compliance_range = [0.0, 1.0]
            randomize_thickness = True
            thickness_range = [0.0, 0.05]

        class dof_properties:
            randomize_stiffness = True
            added_stiffness_range = [-5.0, 5.0]
            randomize_damping = True
            added_damping_range = [-.2, .2]

    class rewards(LeggedRobotTrajectoryCfg.rewards):
        class scales(LeggedRobotTrajectoryCfg.rewards.scales):
            orientation = -1.
            collision = -1.
            action_rate = -0.1
            torques = -0.00001
            # dof_acc = -2.5e-7

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = .55
        max_contact_force = 100.  # forces above this value are penalized


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