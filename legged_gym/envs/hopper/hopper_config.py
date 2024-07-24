from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HopperRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 145
        num_actions = 4  # Changes based on control type

    class terrain( LeggedRobotCfg.terrain):
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        mesh_type = 'plane'

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, .3]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'foot_slide': 0.0,
            'wheel1_rotation': 0.0,
            'wheel2_rotation': 0.0,
            'wheel3_rotation': 0.0
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            'foot_slide': 400,
            'wheel1_rotation': 90.0,
            'wheel2_rotation': 90.0,
            'wheel3_rotation': 90.0
        }  # [N*m/rad for revolute, N/m for prismatic]

        damping = {
            'foot_slide': 40,
            'wheel1_rotation': 7.0,
            'wheel2_rotation': 7.0,
            'wheel3_rotation': 7.0
        }  # [N*m*s/rad for revolute, N*s/m for prismatic]

        wheel_spindown = {
            'wheel1_rotation': 0.1,
            'wheel2_rotation': 0.1,
            'wheel3_rotation': 0.1
        }  # [N*m/rad for revolute, N/m for prismatic]

        foot_pos_des = 0.05

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        control_type = "orientation"
        # control_type = "orientation_spindown"

        zero_action = [1.0, 0, 0, 0]

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hopper/urdf/hopper.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hopper/urdf/hopper_axis_aligned.urdf'
        name = "hopper"
        foot_name = 'foot'
        terminate_after_contacts_on = ['wheel', 'torso']
        flip_visual_attachments = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

        # Foot spring properties
        spring_stiffness = 10000
        spring_damping = 100

        # quat_actuator = [0.8806, 0.3646, -0.2795, 0.1160]
        # rot_actuator = [
        #     [0.9429, 0, 0.3986],
        #     [-0.4507, 0.6622, 0.6374],
        #     [-0.4507, -0.6622, 0.6374]
        # ]
        rot_actuator = [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]
        wheel_speed_bounds = {
            'wheel1_rotation': 600,
            'wheel2_rotation': 600,
            'wheel3_rotation': 600
        }
        torque_speed_bound_ratio = 6
        disable_gravity = False

    class commands:
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.35, 0.35] # min max [m/s]
            lin_vel_y = [-0.35, 0.35]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            orientation = -1.
            collision = -1.
            action_rate = -0.01

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = .55
        max_contact_force = 100.  # forces above this value are penalized


class HopperRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'hopper'
        load_run = -1
        max_iterations = 300
