from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AdamRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4098
        num_observations = 175
        num_actions = 14

    class terrain( LeggedRobotCfg.terrain):
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, .5]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': 0.0,
            'left_knee_pitch_joint': 0.0,

            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': 0.0,
            'right_knee_pitch_joint': 0.0,

            'left_shoulder_yaw_joint': 0.0,
            'left_shoulder_pitch_joint': 0.0,
            'left_forearm_pitch_joint': 0.0,

            'right_shoulder_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'right_forearm_pitch_joint': 0.0
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            'left_hip_yaw_joint': 100.0, 'left_hip_roll_joint': 100.0,
            'left_hip_pitch_joint': 200.0, 'left_knee_pitch_joint': 200.0,
            'right_hip_yaw_joint': 100.0, 'right_hip_roll_joint': 100.0,
            'right_hip_pitch_joint': 200.0, 'right_knee_pitch_joint': 200.0,
            'left_shoulder_yaw_joint': 50.0, 'left_shoulder_pitch_joint': 60.0,
            'left_forearm_pitch_joint': 40.0,
            'right_shoulder_yaw_joint': 50.0, 'right_shoulder_pitch_joint': 60.0,
            'right_forearm_pitch_joint': 40.0
        }  # [N*m/rad]
        damping = {
            'left_hip_yaw_joint': 3.0, 'left_hip_roll_joint': 3.0,
            'left_hip_pitch_joint': 6.0, 'left_knee_pitch_joint': 6.0,
            'right_hip_yaw_joint': 3.0, 'right_hip_roll_joint': 3.0,
            'right_hip_pitch_joint': 6.0, 'right_knee_pitch_joint': 6.0,
            'left_shoulder_yaw_joint': 2.0, 'left_shoulder_pitch_joint': 3.0,
            'left_forearm_pitch_joint': 2.0,
            'right_shoulder_yaw_joint': 2.0, 'right_shoulder_pitch_joint': 3.0,
            'right_forearm_pitch_joint': 2.0
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/adam/urdf/adam.urdf'
        name = "adam"
        foot_name = 'foot'
        penalize_contacts_on = ["hip", "shin", 'hand']
        terminate_after_contacts_on = ['torso', 'shoulder', 'forearm']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 5.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.

class AdamRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'adam'
        load_run = -1
        max_iterations = 300
