from deep_tube_learning.data_collection import data_creation_main

import torch
import wandb
import pickle
import numpy as np
from scipy.io import savemat
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from deep_tube_learning.utils import wandb_model_load, wandb_load_artifact, unnormalize_dict

save_tag = "os"
"""Assumes all models trained on the same dataset"""
def eval_model(delta_seed=1, num_robots=2):
    # Experiment whose model to evaluate
    # exp_names = [  # Hopper Recursive
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/sjiqi49f",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/7p1zump7",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/msj97p19",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/ks1eg0xw",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/rg5itafm",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/j88i9kim",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/mnfs3r5v",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/r5xu847t",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/l4wnnx72",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/w8flp57h",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/f9zr70ds"
    # ]
    # exp_names = [  # Hopper Oneshot
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/dy9ivoc2",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/kma11ykr",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/i8bev8qs",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/743mwobx",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/q9r1jp40",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/55xbfsa8",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/xstep89m",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/r0hxk04w",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/ho9ss350",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/y7udluc0",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/1rf51hpn"
    # ]
    # exp_names = [  # Double oneshot
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/d2rcffmp",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/zsjzccbb",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/4tcr7hfa",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/6w6s1don",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/bte0x966",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/4zawox51",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/syqvx07n",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/6gn5gan2",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/a65vb9rm",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/adeuul8f",
    #     "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/3iy8gtai"
    # ]
    exp_names = [  # Double recursive
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/mrw2ieou",
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/vugufzon",
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/28fyddeg",
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/bzcnvzc8",
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/nxwo1usp",
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/2p3relpc",
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/umfjf3uu",
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/thsm2yqh",
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/g7yvoy73",
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/wn7ag3y4",
        "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/42cp4zi6"
    ]

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, f'{exp_names[0]}_model:best')

    run_id = model_cfg.dataset.wandb_experiment
    with open(f"../rom_tracking_data/{run_id}/config.pickle", 'rb') as f:
        dataset_cfg = pickle.load(f)
    dataset_cfg = OmegaConf.create(unnormalize_dict(dataset_cfg))
    dataset_cfg.epochs = 1
    dataset_cfg.seed += delta_seed
    dataset_cfg.env_config.env.num_envs = num_robots
    dataset_cfg.upload_to_wandb = False
    dataset_cfg.save_debugging_data = True
    dataset_cfg.env_config.domain_rand.max_rom_dist = [0.1, 0.1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_data = data_creation_main(dataset_cfg)

    mat_data = {}

    for _ in range(10):
        k = torch.randint(50, 175, (1,)).item()
        for exp_name in exp_names:
            # Load the model
            model_name = f'{exp_name}_model:best'
            model_cfg, state_dict = wandb_model_load(api, model_name)
            model_cfg.dataset._target_ = model_cfg.dataset._target_.replace('from_wandb', 'from_dataset')
            model_cfg.dataset._partial_ = True
            del model_cfg.dataset.wandb_experiment
            dataset = instantiate(model_cfg.dataset)(dataset=epoch_data)
            data, target = dataset._get_item_helper(0, model_cfg.dataset.H_rev + k)

            model = instantiate(model_cfg.model)(dataset.input_dim, dataset.output_dim)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            with torch.no_grad():
                w = model(data[None, :].to(device)).cpu().numpy().squeeze()

                mat_data[f"H_{model_cfg.dataset.H_rev}_w"] = w
                mat_data[f"H_{model_cfg.dataset.H_rev}_e"] = target.cpu().numpy().squeeze()
        fn = f"data/multirun_{save_tag}_{k}.mat"
        savemat(fn, mat_data)
        print(f"Complete!: saved to {fn}")


if __name__ == "__main__":
    eval_model()
