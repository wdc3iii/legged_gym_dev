from deep_tube_learning.data_collection import data_creation_main

import torch
import wandb
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from hydra.utils import instantiate
from deep_tube_learning.utils import unnormalize_dict


def wandb_model_load(api, artifact_name):
    config, artifact = wandb_load_artifact(api, artifact_name)

    dir_name = artifact.download(root=Path("/tmp/wandb_downloads"))
    state_dict = torch.load(str(Path(dir_name) / "model.pth"))
    return config, state_dict


def wandb_load_artifact(api, artifact_name):
    artifact = api.artifact(artifact_name)
    run = artifact.logged_by()
    config = run.config
    config = unnormalize_dict(config)
    config = OmegaConf.create(config)

    return config, artifact


def eval_model():
    # Experiment whose model to evaluate
    # exp_name = f"coleonguard-Georgia Institute of Technology/Deep_Tube_Training/l1hjebct"
    exp_name = f"coleonguard-Georgia Institute of Technology/Deep_Tube_Training/xny38oqa"
    model_name = f'{exp_name}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)
    print(f"alpha: {model_cfg.loss.alpha}")

    dataset = instantiate(model_cfg.dataset)
    model = instantiate(model_cfg.model)(dataset.input_dim, dataset.output_dim)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data_name = f"coleonguard-Georgia Institute of Technology/RoM_Tracking_Data/{model_cfg.dataset.wandb_experiment}:latest"
    data_cfg, _ = wandb_load_artifact(api, data_name)
    data_cfg.seed = 0

    n_robots = 10
    data_cfg.epochs = 1
    data_cfg.num_robots = n_robots
    data_cfg.sample_hold_dt.n_robots = n_robots
    data_cfg.reduced_order_model.n_robots = n_robots
    data_cfg.reduced_order_model.seed = data_cfg.seed
    data_cfg.upload_to_wandb = False
    data_cfg.save_debugging_data = True

    epoch_data = data_creation_main(data_cfg)
    rom = instantiate(data_cfg.reduced_order_model)

    with torch.no_grad():
        succ_rate_total = 0
        for ii in range(n_robots):
            z = epoch_data['z'][:-1, ii, :]
            pz_x = epoch_data['pz_x'][:-1, ii, :]
            v = epoch_data['v'][:, ii, :]
            w = np.linalg.norm(z - epoch_data['pz_x'][:-1, ii, :], axis=-1)
            w_p1 = w.copy()
            w_p1[1:] = w[:-1]

            data = torch.from_numpy(np.hstack((w[:, None], z, v))).float().to(device)
            fw_p1 = model(data).cpu().numpy()
            fw_p1 = np.vstack([np.array([[0]]), fw_p1[:-1]])

            plt.figure()
            plt.plot(w_p1)
            plt.plot(fw_p1)
            plt.legend(['w', 'fw'])
            plt.xlabel('Time')
            plt.ylabel('Tube Size')
            plt.show()

            plt.figure()
            err = np.squeeze(fw_p1) - w_p1
            succ_rate = np.mean(err >= 0)
            succ_rate_total += succ_rate
            print(succ_rate)
            plt.plot(err)
            plt.axhline(0, c='k')
            plt.xlabel('Time')
            plt.ylabel('Tube Error')
            plt.show()

            fig, ax = plt.subplots()
            rom.plot_tube(ax, z[::10], fw_p1[::10])
            rom.plot_spacial(ax, z[::10], 'k.-')
            rom.plot_spacial(ax, pz_x[::10], 'b.-')
            plt.axis("square")
            plt.show()
        print(f"Total Success Rate: {succ_rate_total / n_robots}")


if __name__ == "__main__":
    eval_model()
