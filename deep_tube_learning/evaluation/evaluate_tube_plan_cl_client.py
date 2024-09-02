import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import socket
import struct
import pickle
import wandb
from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from deep_tube_learning.utils import update_args_from_hydra, update_cfgs_from_hydra, wandb_model_load, update_hydra_cfg, unnormalize_dict
from deep_tube_learning.controllers import RaibertHeuristic
from trajopt.rom_dynamics import SingleInt2D

prob_str = 'right'
# prob_str = 'gap'
H = 150

def get_send(client_socket, server_address):
    def send_pz_x(pz_x):
        # print("Sending pz_x: ", pz_x)
        data = struct.pack('!ff', *pz_x.squeeze().tolist())
        client_socket.sendto(data, server_address)
    return send_pz_x


def get_receive(client_socket, device):
    def recieve_z_v():
        data, _ = client_socket.recvfrom(17)
        terminated = struct.unpack('!B', data[0:1])[0]
        zx, zy, vx, vy = struct.unpack('!ffff', data[1:17])
        # print("Received: ", zx, zy, vx, vy)
        return torch.tensor([[zx, zy]], device=device), torch.tensor([[vx, vy]], device=device), terminated
    return recieve_z_v


def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('127.0.0.1', 12345)
    send_pz_x = get_send(client_socket, server_address)

    client_socket.sendto(prob_str.encode(), server_address)
    client_socket.sendto(struct.pack('!I', H), server_address)
    nn_path, _ = client_socket.recvfrom(1024)
    nn_path = nn_path.decode()
    print("nn_path", nn_path)

    # Grab relevant data
    model_name = f'{nn_path}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)

    run_id = model_cfg.dataset.wandb_experiment
    with open(f"../rom_tracking_data/{run_id}/config.pickle", 'rb') as f:
        dataset_cfg = pickle.load(f)
    dataset_cfg = unnormalize_dict(dataset_cfg)
    dataset_cfg['env_config']['env']['num_envs'] = 1

    dataset_cfg['env_config']['domain_rand']['randomize_rom_distance'] = False

    cfg = OmegaConf.create(unnormalize_dict(dataset_cfg))
    if cfg.controller.type == 'rl':
        exp_name = cfg.wandb_experiment
        model_name = f'{exp_name}_model:best{cfg.curriculum}'
        api = wandb.Api()
        rl_cfg, state_dict = wandb_model_load(api, model_name)
    elif cfg.controller.type == 'rh':
        cfg_dir = str(Path(__file__).resolve().parents[1] / "configs" / "rl")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.2"):
            rl_cfg = compose(config_name=cfg.controller.config_name)
    else:
        raise ValueError(f"Controller type {cfg.controller.type} not implemented.")
    rl_cfg = update_hydra_cfg(cfg, rl_cfg)

    args = get_args()
    args = update_args_from_hydra(rl_cfg, args)
    env_cfg, train_cfg = task_registry.get_cfgs(rl_cfg.task)
    env_cfg, train_cfg = update_cfgs_from_hydra(rl_cfg, env_cfg, train_cfg)

    env, env_cfg = task_registry.make_env(name=rl_cfg.task, args=args, env_cfg=env_cfg)
    receive_cmd = get_receive(client_socket, env.device)

    if cfg.controller.type == 'rl':
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                              train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)
    elif cfg.controller.type == 'rh':
        raibert = RaibertHeuristic(cfg)
        policy = raibert.get_inference_policy(device=env.device)
    else:
        raise ValueError(f"IsaacGym controller type {cfg.controller.type} not implemented.")

    terminated = False
    ii = 0

    # Initialization
    env.reset()
    obs = env.get_observations()
    base = torch.clone(env.root_states.detach())
    pz_x = env.rom.proj_z(base)

    # Loop over time steps
    while not terminated:
        # get obs from thing
        send_pz_x(pz_x)
        z_des, v_des, terminated = receive_cmd()

        if 'type' in cfg.controller.keys() and cfg.controller.type == 'rh':
            if isinstance(env.traj_gen.rom, SingleInt2D):
                current_velocity = env.root_states[:, 7:9]
                current_position = env.root_states[:, :2]

                positional_error = z_des - current_position
                quaternion = env.base_quat  # x,y,z,w
                obs = torch.cat((positional_error, current_velocity, v_des, quaternion), dim=1)
        actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        # Save Data
        base = torch.clone(env.root_states.detach())
        pz_x = env.rom.proj_z(base)

        ii += 1

    print("Closing Client Socket...")
    client_socket.close()

if __name__ == '__main__':
    main()
