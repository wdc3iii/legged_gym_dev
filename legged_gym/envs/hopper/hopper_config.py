from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HopperRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 169
        num_actions = 4

    class terrain( LeggedRobotCfg.terrain):
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'foot_slide': 0.0,
            'wheel1_rotation': 0.0,
            'wheel2_rotation': 0.0,
            'wheel3_rotation': 0.0
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            'foot_slide': 100.0,
            'wheel1_rotation': 100.0,
            'wheel2_rotation': 100.0,
            'wheel3_rotation': 100.0
        }  # [N*m/rad for revolute, N/m for prismatic]

        damping = {
            'foot_slide': 3.0,
            'wheel1_rotation': 3.0,
            'wheel2_rotation': 3.0,
            'wheel3_rotation': 3.0
        }  # [N*m*s/rad for revolute, N*s/m for prismatic]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hopper/urdf/hopper.urdf'
        name = "hopper"
        foot_name = 'foot'
        terminate_after_contacts_on = ['wheel', 'torso']
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
