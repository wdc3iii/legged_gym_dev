import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from deep_tube_learning.utils import wandb_model_load
from deep_tube_learning.simple_data_collection import main
from trajopt.rom_dynamics import SingleInt2D


def eval_model():
    # Experiment whose model to evaluate
    exp_name = f"coleonguard-Georgia Institute of Technology/Deep_Tube_Training/yt4zlljz"
    model_name = f'{exp_name}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)
    model = instantiate(model_cfg.model)(5, 1)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"alpha: {model_cfg.loss.alpha}")

    n_robots = 1
    epoch_data = main(n_robots, 1)

    with torch.no_grad():
        succ_rate_single_total, succ_rate_total = 0, 0
        for ii in range(n_robots):
            z = epoch_data['z'][:-1, ii, :]
            pz_x = epoch_data['pz_x'][:-1, ii, :]
            v = epoch_data['v'][:, ii, :]
            w = np.linalg.norm(pz_x - z, axis=-1)
            w_p1 = w.copy()
            fw = w.copy()
            w_p1[1:] = w[:-1]

            for t in range(w.shape[0] - 1):
                data = torch.from_numpy(np.hstack((fw[t], z[t, :], v[t, :]))).float().to(device)
                fw[t + 1] = model(data).cpu().numpy()

            data = torch.from_numpy(np.hstack((w[:, None], z, v))).float().to(device)
            fw_single = model(data).cpu().numpy()
            fw_single = np.vstack([np.array([[0]]), fw_single[:-1]])

            plt.figure()
            plt.plot(w_p1)
            plt.plot(fw)
            plt.legend(['w', 'fw'])
            plt.xlabel('Time')
            plt.ylabel('Horizon Tube Size')
            plt.show()

            plt.figure()
            plt.plot(w_p1)
            plt.plot(fw_single)
            plt.legend(['w', 'fw_single'])
            plt.xlabel('Time')
            plt.ylabel('Single Tube Size')
            plt.show()

            plt.figure()
            err = np.squeeze(fw) - w_p1
            succ_rate = np.mean(err >= 0)
            succ_rate_total += succ_rate
            print(succ_rate)
            plt.plot(err)
            plt.axhline(0, c='k')
            plt.xlabel('Time')
            plt.ylabel('Tube Error')
            plt.show()

            plt.figure()
            err = np.squeeze(fw_single) - w_p1
            succ_rate_single = np.mean(err >= 0)
            succ_rate_single_total += succ_rate_single
            print(succ_rate_single)
            plt.plot(err)
            plt.axhline(0, c='k')
            plt.xlabel('Time')
            plt.ylabel('Single Tube Error')
            plt.show()

            fig, ax = plt.subplots()
            SingleInt2D.plot_tube(ax, z[::2], fw[::2, None])
            SingleInt2D.plot_spacial(ax, z[::2], 'k.-')
            SingleInt2D.plot_spacial(ax, pz_x[::2], 'b.-')
            plt.axis("square")
            plt.title("Horizon Tube")
            plt.show()

            fig, ax = plt.subplots()
            SingleInt2D.plot_tube(ax, z[::2], fw_single[::2])
            SingleInt2D.plot_spacial(ax, z[::2], 'k.-')
            SingleInt2D.plot_spacial(ax, pz_x[::2], 'b.-')
            plt.axis("square")
            plt.title("Single Tube")
            plt.show()
        print(f"Total Success Rate: {succ_rate_total / n_robots}")


if __name__ == "__main__":
    eval_model()
