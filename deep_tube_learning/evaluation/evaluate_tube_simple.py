import wandb
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from deep_tube_learning.utils import wandb_model_load
from deep_tube_learning.simple_data_collection import main
from deep_tube_learning.datasets import sliding_window
from trajopt.rom_dynamics import SingleInt2D
import torch


def eval_model():
    # Experiment whose model to evaluate
    # exp_name = f"coleonguard-Georgia Institute of Technology/Deep_Tube_Training/eappxsy7"  # Standard
    # exp_name = f"coleonguard-Georgia Institute of Technology/Deep_Tube_Training/mm9v4zyv"  # Input History
    exp_name = f"coleonguard-Georgia Institute of Technology/Deep_Tube_Training/asfxz573"  # Error/Input History
    model_name = f'{exp_name}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)
    N = model_cfg.dataset.N if 'N' in model_cfg.dataset.keys() else 1
    dN = model_cfg.dataset.dN if 'dN' in model_cfg.dataset.keys() else 1
    recursive = model_cfg.dataset.recursive

    n_robots = 2
    epoch_data = main(n_robots, 1)

    z = epoch_data['z'][:, :-1, :]
    pz_x = epoch_data['pz_x'][:, :-1, :]
    v = epoch_data['v']
    w = np.linalg.norm(pz_x - z, axis=-1)
    w_p1 = w.copy()
    w_p1[:, 1:] = w[:, :-1]

    z_no_pos = z[:, :, 2:]

    if recursive:
        data = np.concatenate((w[:, :, None], z_no_pos, v), axis=-1)
        rep_dim = data.shape[-1]
        window_data = sliding_window(data, N, dN, v.shape[-1])
    else:
        zv = np.concatenate((z_no_pos, v), axis=-1)
        rep_dim = zv.shape[-1]
        zv_slide = sliding_window(zv, N, dN, v.shape[-1])
        window_data = np.concatenate((w[:, :, None], zv_slide), axis=-1)

    model = instantiate(model_cfg.model)(window_data.shape[-1], 1)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"alpha: {model_cfg.loss.alpha}")

    with torch.no_grad():
        succ_rate_single_total, succ_rate_total = 0, 0
        for ii in range(n_robots):
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

            plt.figure()
            plt.plot(w[ii])
            plt.plot(fw)
            plt.legend(['w', 'fw'])
            plt.xlabel('Time')
            plt.ylabel('Horizon Tube Size')
            plt.show()

            plt.figure()
            plt.plot(w[ii])
            plt.plot(fw_single)
            plt.legend(['w', 'fw_single'])
            plt.xlabel('Time')
            plt.ylabel('Single Tube Size')
            plt.show()

            plt.figure()
            err = np.squeeze(fw) - w[ii]
            succ_rate = np.mean(err >= 0)
            succ_rate_total += succ_rate
            print(succ_rate)
            plt.plot(err)
            plt.axhline(0, c='k')
            plt.xlabel('Time')
            plt.ylabel('Tube Error')
            plt.show()

            plt.figure()
            err = np.squeeze(fw_single) - w[ii]
            succ_rate_single = np.mean(err >= 0)
            succ_rate_single_total += succ_rate_single
            print(succ_rate_single)
            plt.plot(err)
            plt.axhline(0, c='k')
            plt.xlabel('Time')
            plt.ylabel('Single Tube Error')
            plt.show()

            plt.figure()
            plt.plot(fw_single, 'k', label='Tube')
            plt.plot(w[ii], 'b', label='Normed Error')
            plt.axhline(0, color='black', linewidth=0.5)
            plt.legend()
            plt.title("Single Tube Bounds")
            plt.show()

            fig, ax = plt.subplots()
            SingleInt2D.plot_tube(ax, z[ii], fw_single[:, None])
            SingleInt2D.plot_spacial(ax, z[ii], 'k.-')
            SingleInt2D.plot_spacial(ax, pz_x[ii], 'b.-')
            plt.axis("square")
            plt.title("Single Tube")
            plt.show()

            plt.figure()
            plt.plot(fw, 'r', label='Tube Error')
            plt.plot(w[ii], 'b', label='Normed Error')
            plt.axhline(0, color='black', linewidth=0.5)
            plt.title("Horizon Tube Bounds")
            plt.ylim([0, 2])
            plt.legend()
            plt.show()

            fig, ax = plt.subplots()
            SingleInt2D.plot_tube(ax, z[ii], fw[:, None])
            SingleInt2D.plot_spacial(ax, z[ii], 'k.-')
            SingleInt2D.plot_spacial(ax, pz_x[ii], 'b.-')
            plt.axis("square")
            plt.title("Horizon Tube")
            plt.show()
        print(f"Total Success Rate: {succ_rate_total / n_robots}")


if __name__ == "__main__":
    eval_model()
