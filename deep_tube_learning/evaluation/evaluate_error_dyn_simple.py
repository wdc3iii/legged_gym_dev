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
    exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/0qtznszg"
    model_name = f'{exp_name}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)
    model = instantiate(model_cfg.model)(6, 2)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    n_robots = 1
    epoch_data = main(n_robots, 1)

    with torch.no_grad():
        err_total, err_os_total = 0, 0
        for ii in range(n_robots):
            z = epoch_data['z'][:-1, ii, :]
            pz_x = epoch_data['pz_x'][:-1, ii, :]
            v = epoch_data['v'][:, ii, :]
            e = pz_x - z

            single_data = torch.from_numpy(np.hstack((e, z, v))).float().to(device)
            fe_single = model(single_data).cpu().numpy()
            fe_single = np.vstack((e[0, :], fe_single[:-1, :]))

            fe = e.copy()
            for t in range(e.shape[0] - 1):
                data = torch.from_numpy(np.hstack((fe[t, :], z[t, :], v[t, :]))).float().to(device)
                fe[t + 1, :] = model(data).cpu().numpy()

            plt.figure()
            plt.plot(e, '--')
            plt.plot(fe)
            plt.xlabel('Time')
            plt.ylabel('Horizon Tracking Error')
            plt.show()

            plt.figure()
            plt.plot(e, '--')
            plt.plot(fe_single)
            plt.xlabel('Time')
            plt.ylabel('One Step Tracking Error')
            plt.show()

            plt.figure()
            err_norm = np.linalg.norm(e - fe, axis=-1)
            err_total += torch.nn.MSELoss()(torch.from_numpy(e), torch.from_numpy(fe))
            print(np.mean(err_norm))
            plt.plot(err_norm)
            plt.plot(np.linalg.norm(e, axis=-1))
            plt.axhline(0, c='k')
            plt.legend(['fe adjusted', 'nominal'])
            plt.xlabel('Time')
            plt.ylabel('fe Horizon Prediction Error')
            plt.show()

            plt.figure()
            err_os_norm = np.linalg.norm(e - fe_single, axis=-1)
            err_os_total += torch.nn.MSELoss()(torch.from_numpy(e), torch.from_numpy(fe_single))
            print(np.mean(err_os_norm))
            plt.plot(err_os_norm)
            plt.plot(np.linalg.norm(e, axis=-1))
            plt.axhline(0, c='k')
            plt.xlabel('Time')
            plt.ylabel('fe One Step Prediction Error')
            plt.legend(['fe adjusted', 'nominal'])
            plt.show()

            fig, ax = plt.subplots()
            SingleInt2D.plot_spacial(ax, z, 'k-')
            SingleInt2D.plot_spacial(ax, pz_x, 'b-')
            SingleInt2D.plot_spacial(ax, z + fe, 'r-')
            plt.axis("square")
            plt.legend(['z', 'Pz_x', 'z + fe'])
            plt.show()

            fig, ax = plt.subplots()
            SingleInt2D.plot_spacial(ax, z, 'k-')
            SingleInt2D.plot_spacial(ax, pz_x, 'b-')
            SingleInt2D.plot_spacial(ax, z + fe_single, 'r-')
            plt.axis("square")
            plt.legend(['z', 'Pz_x', 'z + fe_single'])
            plt.show()
        print(f"Mean Error: {err_total / n_robots}")
        print(f"Mean One Step Error: {err_os_total / n_robots}")


if __name__ == "__main__":
    eval_model()
