import isaacgym
from legged_gym.envs import *
from isaacgym import gymtorch
from legged_gym.utils import get_args, task_registry


import torch
import numpy as np
import pandas as pd
from scipy.io import savemat

EP_LEN = int(600)


def main():
    """main"""
    task = "hopper_flat"

    """First, load in the environment"""
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)

    # Override some parameters for testing
    env_cfg.env.num_envs = 2
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # Prepare environment
    args = get_args()
    args.headless = False
    env, _ = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    """Next, set up the initial condition and controller/control signal"""
    base_quat = np.array([0., 0., 0., 1.])
    # base_quat = np.array([0., 0.0995, 0., 0.9950])
    base_pos = np.array([0., 0., 0.49])
    base_vel = np.zeros((3,))
    base_ang_vel = np.zeros((3,))

    joint_pos = np.zeros((4,))
    joint_vel = np.zeros((4,))


    env.dof_pos[:] = torch.from_numpy(joint_pos).float().to(env.device)
    env.dof_vel[:] = torch.from_numpy(joint_vel).float().to(env.device)
    root_state = np.concatenate((base_pos, base_quat, base_vel, base_ang_vel), axis=0)
    root_state = np.repeat(root_state[None, :], 2, axis=0)
    root_state[1, 0] += 1
    env.root_states[:] = torch.from_numpy(root_state).float().to(env.device)

    env_ids_int32 = torch.arange(2, device=env.device).to(dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(env.sim,
                                          gymtorch.unwrap_tensor(env.dof_state),
                                          gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                 gymtorch.unwrap_tensor(env.root_states),
                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    env.render()

    def control(t, x):
        a = torch.zeros((2, 4))
        a[:, 0] = 1.0
        return a

    cols = [
        't', 'contact',
        'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'foot', 'w1', 'w2', 'w3',
        'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'vfoot', 'vw1', 'vw2', 'vw3',
        'tau_f', 'tau_w1', 'tau_w2', 'tau_w3'
    ]
    trajectory = np.zeros((EP_LEN, len(cols)))

    """Finally, run the simulation"""
    for tt in range(EP_LEN):
        # Create state vector
        base = env.root_states.cpu().numpy()
        joint_pos = env.dof_pos.cpu().numpy()
        joint_vel = env.dof_vel.cpu().numpy()
        x = np.concatenate((base[:, :7], joint_pos, base[:, 7:], joint_vel), axis=1)

        contacts = torch.squeeze(env.contact_forces[:, env.feet_indices, 2] > 0.1)

        trajectory[tt, 0] = env.dt * tt
        trajectory[tt, 1] = contacts[0]
        trajectory[tt, 2:23] = x[0, :]
        trajectory[tt, 23:] = env.torques[0, :].cpu().numpy()

        actions = control(tt, x)
        obs, _, _, dones, _ = env.step(actions.detach())

        if torch.any(dones):
            print("Env Done!")

    data = pd.DataFrame(trajectory, columns=cols)
    savemat('sim2sim.mat', {col: data[col].values for col in data.columns})


if __name__ == "__main__":
    main()
