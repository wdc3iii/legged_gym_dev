from deep_tube_learning.data_collection import data_creation_main

import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from deep_tube_learning.datasets import ScalarHorizonTubeDataset
from deep_tube_learning.utils import wandb_model_load, wandb_load_artifact
from trajopt.rom_dynamics import SingleInt2D


def eval_model():
    # Experiment whose model to evaluate
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/idbt0oad"
    exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/isdo1nyp"
    model_name = f'{exp_name}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)
    H_fwd = model_cfg.dataset.H_fwd
    H_rev = model_cfg.dataset.H_rev

    data_name = f"coleonguard-Georgia Institute of Technology/RoM_Tracking_Data/{model_cfg.dataset.wandb_experiment}:latest"
    data_cfg, _ = wandb_load_artifact(api, data_name)
    data_cfg.seed = 1
    data_cfg.env_config.domain_rand.max_rom_dist = [0., 0.]

    n_robots = 2
    data_cfg.epochs = 1
    data_cfg.upload_to_wandb = False
    data_cfg.save_debugging_data = False

    epoch_data = data_creation_main(data_cfg)
    z = epoch_data['z'][:, :-1, :]
    pz_x = epoch_data['pz_x'][:, :-1, :]
    v = epoch_data['v']
    v = np.concatenate((np.zeros((v.shape[0], H_rev, v.shape[2])), v), axis=1)
    z_rev = np.repeat(z[:, None, 0, :], H_rev, axis=1)  # TODO: zero out non-position components
    z = np.concatenate((z_rev, z), axis=1)
    pz_x_rev = np.repeat(pz_x[:, None, 0, :], H_rev, axis=1)
    pz_x = np.concatenate((pz_x_rev, pz_x), axis=1)
    w = np.linalg.norm(pz_x - z, axis=-1)

    z_no_pos = z[:, :, 2:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ScalarHorizonTubeDataset(
        torch.from_numpy(w).float(),
        torch.from_numpy(z_no_pos).float(),
        torch.from_numpy(v).float(),
        H_fwd, H_rev, H_rev + z_no_pos.shape[-1] + (H_rev + H_fwd) * v.shape[-1], H_fwd
    )
    zero = torch.ones((1,), dtype=torch.int) * H_rev
    tmp_data, tmp_target = dataset._get_item_helper(0, zero)
    model = instantiate(model_cfg.model)(dataset.input_dim, dataset.output_dim)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"alpha: {model_cfg.loss.alpha}")

    rom_cfg = data_cfg.env_config.rom
    model_class = globals()[rom_cfg.cls]
    rom = model_class(
        dt=0.02,
        z_min=torch.tensor(rom_cfg.z_min, device=device),
        z_max=torch.tensor(rom_cfg.z_max, device=device),
        v_min=torch.tensor(rom_cfg.v_min, device=device),
        v_max=torch.tensor(rom_cfg.v_max, device=device),
        n_robots=1,
        backend='torch',
        device=device
    )


    with torch.no_grad():
        succ_rate_single_total, succ_rate_total = 0, 0
        for ii in range(100):
            data, tmp_target = dataset._get_item_helper(0, zero + ii)
            fw = model(data.to(device)).cpu().detach().numpy()
            fw = np.concatenate([w[0, [H_rev + ii]], fw], axis=-1)

            plt.figure()
            err = np.squeeze(fw) - w[0, H_rev + ii:H_rev + H_fwd + 1 + ii]
            succ_rate = np.mean(err >= 0)
            succ_rate_total += succ_rate
            print(succ_rate)
            plt.plot(err)
            plt.axhline(0, c='k')
            plt.xlabel('Time')
            plt.ylabel('Tube Error')
            plt.show()

            plt.figure()
            plt.plot(fw, 'k', label='Tube Error')
            plt.plot(w[0, H_rev + ii:H_rev + H_fwd + 1 + ii], 'b', label='Normed Error')
            plt.axhline(0, color='black', linewidth=0.5)
            plt.title("OneShot Tube Bounds")
            plt.legend()
            plt.show()

            fig, ax = plt.subplots()
            rom.plot_tube(ax, z[0, H_rev + ii:H_rev + H_fwd + 1+ ii], fw[:, None])
            rom.plot_spacial(ax, z[0, H_rev + ii:H_rev + H_fwd + 1 + ii], 'k.-')
            rom.plot_spacial(ax, pz_x[0, H_rev + ii:H_rev + H_fwd + 1 + ii])
            plt.axis("square")
            plt.title("OneShot Tube")
            plt.show()
        print(f"Total Success Rate: {succ_rate_total / n_robots}")


if __name__ == "__main__":
    eval_model()
