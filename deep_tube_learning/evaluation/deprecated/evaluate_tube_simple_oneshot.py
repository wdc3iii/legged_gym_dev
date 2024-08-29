import wandb
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from deep_tube_learning.utils import wandb_model_load
from deep_tube_learning.deprecated.simple_data_collection import main
from deep_tube_learning.datasets import ScalarHorizonTubeDataset
from trajopt.rom_dynamics import SingleInt2D
import torch


def eval_model():
    # Experiment whose model to evaluate
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/l0oh8o1b"  # 32x32
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/k1kfktrl"   # 128x128
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/3vdx800j"   # 256x256
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/trq7kcv2"    # 128x128 Softplus
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/yasik42v"  # 128x128 Softplus b=5
    # exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/t3b8qehd"  # Updated 128x128 Softplus b=5
    exp_name = "coleonguard-Georgia Institute of Technology/Deep_Tube_Training/isdo1nyp"  # 128x128 S+b5, some zero error training data

    model_name = f'{exp_name}_model:best'

    api = wandb.Api()
    model_cfg, state_dict = wandb_model_load(api, model_name)
    H_fwd = model_cfg.dataset.H_fwd
    H_rev = model_cfg.dataset.H_rev

    n_robots = 2
    epoch_data = main(n_robots, 1, max_rom_dist=0.)

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

    with torch.no_grad():
        succ_rate_single_total, succ_rate_total = 0, 0
        for ii in range(n_robots):

            data, tmp_target = dataset._get_item_helper(ii, zero)
            fw = model(data.to(device)).cpu().detach().numpy()
            fw = np.concatenate([w[ii, [H_rev]], fw], axis=-1)

            plt.figure()
            err = np.squeeze(fw) - w[ii, H_rev:H_rev+H_fwd+1]
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
            plt.plot(w[ii, H_rev:H_rev+H_fwd+1], 'b', label='Normed Error')
            plt.axhline(0, color='black', linewidth=0.5)
            plt.title("OneShot Tube Bounds")
            plt.legend()
            plt.savefig('data/OneShotTubeBound.png')
            plt.show()

            fig, ax = plt.subplots()
            SingleInt2D.plot_tube(ax, z[ii, H_rev:H_rev+H_fwd+1], fw[:, None])
            SingleInt2D.plot_spacial(ax, z[ii, H_rev:H_rev+H_fwd+1], 'k.-')
            SingleInt2D.plot_spacial(ax, pz_x[ii, H_rev:H_rev+H_fwd+1], 'b.-')
            plt.axis("square")
            plt.title("OneShot Tube")
            plt.show()
        print(f"Total Success Rate: {succ_rate_total / n_robots}")


if __name__ == "__main__":
    eval_model()
