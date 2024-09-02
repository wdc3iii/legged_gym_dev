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


def eval_model(delta_seed=1, num_robots=2):
    # Experiment whose model to evaluate
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/pl0dhg5j"  # Larger bounds
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/c4izk9vs"  # Tighter bounds
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/002384lb"  # Tightest bounds

    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/b0ein4nu"  # Recursive

    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/0i2o675r"  # Hopper Single Int
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/jtu9xrfq"  # Hopper with ff
    exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/43tiikpa"  # H2H

    model_name = f'{exp_name}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)

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

    dataset = instantiate(model_cfg.dataset)
    model = instantiate(model_cfg.model)(dataset.input_dim, dataset.output_dim)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    epoch_data = data_creation_main(dataset_cfg)

    model_cfg.dataset._target_ = model_cfg.dataset._target_.replace('from_wandb', 'from_dataset')
    model_cfg.dataset._partial_ = True
    del model_cfg.dataset.wandb_experiment
    dataset = instantiate(model_cfg.dataset)(dataset=epoch_data)
    with torch.no_grad():
        for ii in range(dataset_cfg.env_config.env.num_envs):
            tmp_data, tmp_target = dataset._get_item_helper(ii, dataset.H_rev)
            z = epoch_data['z'][ii]
            v = epoch_data['v'][ii]
            pz_x = epoch_data['pz_x'][ii]
            H = v.shape[0] - dataset.H_fwd
            w = np.zeros((H, *tmp_target.shape))
            e = np.linalg.norm(pz_x - z, axis=-1)
            for k in range(H):
                data, ek = dataset._get_item_helper(ii, k + dataset.H_rev)
                w[k, :] = model(data[None, :].to(device)).cpu().numpy()

            fn = f"data/eval_{run_id[:-8] + exp_name[-8:]}_{ii}.mat"
            savemat(fn, {
                "z": z,
                "v": v,
                "e": e,
                "w": w,
                "pz_x": pz_x,
            })
        print(f"Complete!: saved to {fn}")


if __name__ == "__main__":
    eval_model()
