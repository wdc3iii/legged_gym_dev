import wandb
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from deep_tube_learning.utils import wandb_model_load
from deep_tube_learning.deprecated.simple_data_collection import main
from deep_tube_learning.datasets import sliding_window, ScalarHorizonTubeDataset
import torch


def eval_model():
    # Experiment whose model to evaluate
    exp_names = ["coleonguard-Georgia Institute of Technology/Deep_Tube_Training/eappxsy7",  # Standard
                 "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/mm9v4zyv",  # Input History
                 "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/asfxz573",  # Error/Input History
                 "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/3vdx800j"]   # 256x256
    model_names = [f'{exp_name}_model:best' for exp_name in exp_names]
    tags = ['N1 Tube', 'N10 Tube', 'N10_r Tube', 'OneShot']

    api = wandb.Api()
    cfg_dicts = [wandb_model_load(api, model_name) for model_name in model_names]

    n_robots = 2
    epoch_data = main(n_robots, 1, max_rom_dist=0.)

    z = epoch_data['z'][:, :-1, :]
    pz_x = epoch_data['pz_x'][:, :-1, :]
    v = epoch_data['v']
    w = np.linalg.norm(pz_x - z, axis=-1)
    w_p1 = w.copy()
    w_p1[:, 1:] = w[:, :-1]

    z_no_pos = z[:, :, 2:]

    def get_data_and_model(model_cfg, state_dict):
        N = model_cfg.dataset.N if 'N' in model_cfg.dataset.keys() else 1
        dN = model_cfg.dataset.dN if 'dN' in model_cfg.dataset.keys() else 1
        recursive = model_cfg.dataset.recursive if 'recursive' in model_cfg.dataset.keys() else False
        H = model_cfg.dataset.H if 'H' in model_cfg.dataset.keys() else None
        if recursive:
            data = np.concatenate((w[:, :, None], z_no_pos, v), axis=-1)
            rep_dim = data.shape[-1]
            window_data = sliding_window(data, N, dN, v.shape[-1])
        else:
            zv = np.concatenate((z_no_pos, v), axis=-1)
            rep_dim = zv.shape[-1]
            zv_slide = sliding_window(zv, N, dN, v.shape[-1])
            window_data = np.concatenate((w[:, :, None], zv_slide), axis=-1)
        if H is None:
            model = instantiate(model_cfg.model)(window_data.shape[-1], 1)

        else:
            model = instantiate(model_cfg.model)(1 + z_no_pos.shape[-1] + H * v.shape[-1], H)
            window_data = None
        model.load_state_dict(state_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return window_data, model, recursive, rep_dim

    datas_models = [get_data_and_model(model_cfg, state_dict) for model_cfg, state_dict in cfg_dicts]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = 50
    oneshot_dataset = ScalarHorizonTubeDataset(
        torch.from_numpy(w).float(),
        torch.from_numpy(z_no_pos).float(),
        torch.from_numpy(v).float(),
        H, 1 + z_no_pos.shape[-1] + H * v.shape[-1], H
    )

    with torch.no_grad():
        succ_rate_single_total, succ_rate_total = 0, 0
        for ii in range(n_robots):
            print(ii)
            fws = []
            fw_singles = []
            for window_data, model, recursive, rep_dim in datas_models:
                if window_data is not None:
                    single_data = window_data[ii]
                    fw_single = model(torch.from_numpy(single_data).float().to(device)).cpu().detach().numpy()
                    fw_single = np.vstack((w[ii, 0], fw_single[:-1, :]))

                    fw = w[ii].copy()
                    data = single_data[0]
                    for t in range(w.shape[1] - 1):
                        fw[t + 1] = model(torch.from_numpy(data).float().to(device)).cpu().detach().numpy()
                        if recursive:
                            data[rep_dim:] = data[:-rep_dim]
                            data[0] = fw[t + 1]
                            data[1:rep_dim] = single_data[t + 1, 1:rep_dim]
                        else:
                            data = single_data[t + 1]
                            data[0] = fw[t + 1]
                    fw_singles.append(fw_single)
                    fws.append(fw)
                else:
                    fw = w[ii].copy()
                    fw[1:] = np.nan
                    # for jj in range(0, z.shape[1] - H, H):
                    for jj in range(0, 1, H):
                        ind = jj * torch.zeros((1,), dtype=torch.int)
                        data, tmp_target = oneshot_dataset._get_item_helper(ii, ind)
                        fw[jj + 1:jj + H + 1] = model(data.to(device)).cpu().detach().numpy()
                    fws.append(fw)

            plt.figure()
            plt.plot(w[ii], 'k', label='Normed Error')
            for fw_single, tag in zip(fw_singles, tags):
                plt.plot(fw_single, label=tag)
            plt.axhline(0, color='black', linewidth=0.5)
            plt.legend()
            plt.title("Single Tube Bounds")
            plt.savefig("data/SingeTubeBounds.png")
            plt.show()

            plt.figure()
            plt.plot(w[ii], 'k', label='Normed Error')
            for fw, tag in zip(fws, tags):
                plt.plot(fw, label=tag)
            plt.axhline(0, color='black', linewidth=0.5)
            plt.legend()
            plt.title("Horizon Tube Bounds")
            plt.ylim([0, 2])
            plt.savefig("data/HorizonTubeBounds.png")
            plt.show()
        print(f"Total Success Rate: {succ_rate_total / n_robots}")


if __name__ == "__main__":
    eval_model()
