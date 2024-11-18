import torch
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat


def evaluate_single_double_easy(x, h, delta, save_eval_data, data_path, ii, policy):
    if save_eval_data:
        with open(f"{data_path}/eval_{ii}.pickle", "wb") as f:
            epoch_data = {
                'x': x.cpu().numpy(),
                'h': h.cpu().numpy(),
                'delta': delta.cpu().numpy(),
            }
            pickle.dump(epoch_data, f)
            savemat(f"{data_path}/eval_{ii}.mat", epoch_data)

    print(f"\nLearning Epoch {ii}:\n\tViolation Proportion: {torch.mean((h < 0).float()):.3f}\n\tTraj Violation Prop: {torch.mean(torch.less(torch.min(h, dim=-1)[0], 0).float()):.3f}\n\tMean Violation: {-torch.mean(h[h < 0]):.2e}\n\tMaximum Violation: {max(0, -torch.min(h)):.3f}")
    fig, ax = plt.subplots()
    ax.plot(h[::20, :].cpu().numpy().T)
    ax.axhline(y=0, color='k')
    ax.set_xlabel('Time')
    ax.set_ylabel('h')
    plt.title(f"Evaluation of Training Iteration {ii}")
    plt.show()

    x_ = torch.linspace(0, 1, 100)  # 100 points in the range [0, 1]
    y_ = torch.linspace(-1, 1, 100)  # 100 points in the range [-1, 1]
    xx, yy = torch.meshgrid(x_, y_, indexing='ij')  # Create a grid
    grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2).to("cuda:0")  # Shape: (10000, 2)

    # Pass the grid through the neural network
    with torch.no_grad():
        outputs = policy.v_filt.delta(grid).reshape(100, 100)  # Reshape to 100x100

    # Plot the output
    plt.figure(figsize=(8, 6))
    plt.contourf(xx.cpu().numpy(), yy.cpu().numpy(), outputs.cpu().numpy(), levels=50, cmap='viridis')
    plt.colorbar(label='delta')
    plt.xlabel('x')
    plt.ylabel('dot x')
    plt.title(f"Evaluation of Training Iteration {ii}")
    plt.show()


def evaluate_single_double(x, h, delta, save_eval_data, data_path, ii, policy):
    if save_eval_data:
        with open(f"{data_path}/eval_{ii}.pickle", "wb") as f:
            epoch_data = {
                'x': x.cpu().numpy(),
                'h': h.cpu().numpy(),
                'delta': delta.cpu().numpy(),
            }
            pickle.dump(epoch_data, f)
            savemat(f"{data_path}/eval_{ii}.mat", epoch_data)

    print(f"\nLearning Epoch {ii}:\n\tViolation Proportion: {torch.mean((h < 0).float()):.3f}\n\tTraj Violation Prop: {torch.mean(torch.less(torch.min(h, dim=-1)[0], 0).float()):.3f}\n\tMean Violation: {-torch.mean(h[h < 0]):.2e}\n\tMaximum Violation: {max(0, -torch.min(h)):.3f}")

    valid_inds = h[:, 0] >= 0
    h = h[valid_inds, :]
    x = x[valid_inds, :, :]

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(h[::20, :].cpu().numpy().T)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('H')
    ax[0].axhline(y=0, color='k')

    ax[1].plot(delta[::20, :].cpu().numpy().T)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('delta')
    plt.title(f"Evaluation of Training Iteration {ii}")
    plt.show()

    fig, ax = plt.subplots()
    for i in range(policy.v_filt.cbf.rs.shape[0]):
        circ = plt.Circle((policy.v_filt.cbf.cs[i, 0].item(), policy.v_filt.cbf.cs[i, 1].item()), policy.v_filt.cbf.rs[i].item())
        ax.add_patch(circ)
    ax.plot(x[::20, :, 0].cpu().numpy().T, x[::20, :, 1].cpu().numpy().T)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    plt.title(f"Evaluation of Training Iteration {ii}")
    plt.show()