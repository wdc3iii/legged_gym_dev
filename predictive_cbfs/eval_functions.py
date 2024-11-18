import torch
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat


def evaluate_single_double(policy, env, eval_states, cfg, data_path, ii):
    num_robots = cfg.env_config.env.num_envs
    max_rom_ep_length = int(cfg.env_config.env.episode_length_s / env.dt) - 5

    obs, _ = env.reset()
    obs, _ = env.set_states(torch.clone(eval_states.detach()))
    x_n = obs.shape[1]
    x = torch.zeros((num_robots, max_rom_ep_length + 1, x_n), device=env.device)  # Epochs, steps, states
    h = torch.zeros((num_robots, max_rom_ep_length + 1), device=env.device)
    delta = torch.zeros((num_robots, max_rom_ep_length + 1), device=env.device)
    x[:, 0, :] = obs.detach()
    h[:, 0] = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[:, 0, :]))

    # Loop over time steps
    for t in range(max_rom_ep_length):
        actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        # Save Data
        x[:, t + 1, :] = obs.detach()
        h[:, t + 1] = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[:, t + 1, :]))
        delta[:, t + 1] = policy.v_filt.delta_val.detach()
    with open(f"{data_path}/eval_{ii}.pickle", "wb") as f:
        epoch_data = {
            'x': x.cpu().numpy(),
            'h': h.cpu().numpy(),
            'delta': delta.cpu().numpy(),
        }
        pickle.dump(epoch_data, f)
        savemat(f"{data_path}/eval_{ii}.mat", epoch_data)

    print(f"Maximum Violation: {max(0, -torch.min(h))}\nViolation Proportion: {torch.mean((h < 0).float())}")

    valid_inds = h[:, 0] >= 0
    h = h[valid_inds, :]
    x = x[valid_inds, :, :]

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(h[::20, :].cpu().numpy().T)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('H')

    ax[1].plot(delta[::20, :].cpu().numpy().T)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('delta')
    plt.title(f"Evaluation of Training Iteration {ii}")
    plt.show()

    fig, ax = plt.subplots()
    for i in range(policy.v_filt.cbf.rs.shape[0]):
        circ = plt.Circle((policy.v_filt.cbf.cs[i, 0].item(), policy.v_filt.cbf.cs[i, 1].item()),
                          policy.v_filt.cbf.rs[i].item())
        ax.add_patch(circ)
    ax.plot(x[::20, :, 0].cpu().numpy().T, x[::20, :, 1].cpu().numpy().T)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    plt.title(f"Evaluation of Training Iteration {ii}")
    plt.show()


def evaluate_single_double_easy(policy, env, eval_root_states, cfg, data_path, ii):
    num_robots = cfg.env_config.env.num_envs
    max_rom_ep_length = int(cfg.env_config.env.episode_length_s / env.model.dt) - 5

    env.reset()
    env.root_states = torch.clone(eval_root_states.detach())
    obs = env.get_observations()
    x_n = obs.shape[1]
    x = torch.zeros((num_robots, max_rom_ep_length + 1, x_n), device=env.device)  # Epochs, steps, states
    h = torch.zeros((num_robots, max_rom_ep_length + 1), device=env.device)
    delta = torch.zeros((num_robots, max_rom_ep_length + 1), device=env.device)
    x[:, 0, :] = env.get_states()
    h[:, 0] = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[:, 0, :]))

    # Loop over time steps
    for t in range(max_rom_ep_length):
        actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        # Save Data
        x[:, t + 1, :] = obs
        h[:, t + 1] = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[:, t + 1, :]))
        delta[:, t + 1] = policy.v_filt.delta_val.detach()
    with open(f"{data_path}/eval_{ii}.pickle", "wb") as f:
        epoch_data = {
            'x': x.cpu().numpy(),
            'h': h.cpu().numpy(),
            'delta': delta.cpu().numpy(),
        }
        pickle.dump(epoch_data, f)
        savemat(f"{data_path}/eval_{ii}.mat", epoch_data)

    print(f"Maximum Violation: {max(0, -torch.min(h))}\nViolation Proportion: {torch.mean((h < 0).float())}\nMean Violation: {torch.mean(h[h < 0])}")

    fig, ax = plt.subplots()
    ax.plot(h[::20, :].cpu().numpy().T)
    ax.set_xlabel('Time')
    ax.set_ylabel('h')
    plt.title(f"Evaluation of Training Iteration {ii}")
    plt.show()

    x_ = torch.linspace(0, 1, 100)  # 100 points in the range [0, 1]
    y_ = torch.linspace(-1, 1, 100)  # 100 points in the range [-1, 1]
    xx, yy = torch.meshgrid(x_, y_, indexing='ij')  # Create a grid
    grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2).to(env.device)  # Shape: (10000, 2)

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


def evaluate_single_hopper(policy, env, eval_states, cfg, data_path, ii):
    num_robots = cfg.env_config.env.num_envs
    max_rom_ep_length = int(cfg.env_config.env.episode_length_s / env.dt) - 5

    # Haven't figured out how to set eval_states without destroying everything
    obs, _ = env.reset()
    x_n = obs.shape[1]
    x = torch.zeros((num_robots, max_rom_ep_length + 1, x_n), device=env.device)  # Epochs, steps, states
    h = torch.zeros((num_robots, max_rom_ep_length + 1), device=env.device)
    delta = torch.zeros((num_robots, max_rom_ep_length + 1), device=env.device)
    x[:, 0, :] = obs.detach()
    h[:, 0] = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[:, 0, :]))

    # Loop over time steps
    for t in range(max_rom_ep_length):
        actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        # Save Data
        x[:, t + 1, :] = obs.detach()
        h[:, t + 1] = policy.v_filt.cbf.h(policy.v_filt.dyn.proj(x[:, t + 1, :]))
        delta[:, t + 1] = policy.v_filt.delta_val.detach()
    with open(f"{data_path}/eval_{ii}.pickle", "wb") as f:
        epoch_data = {
            'x': x.cpu().numpy(),
            'h': h.cpu().numpy(),
            'delta': delta.cpu().numpy(),
        }
        pickle.dump(epoch_data, f)
        savemat(f"{data_path}/eval_{ii}.mat", epoch_data)

    print(f"Maximum Violation: {max(0, -torch.min(h))}\nViolation Proportion: {torch.mean((h < 0).float())}")

    valid_inds = h[:, 0] >= 0
    h = h[valid_inds, :]
    x = x[valid_inds, :, :]

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(h[::20, :].cpu().numpy().T)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('H')

    ax[1].plot(delta[::20, :].cpu().numpy().T)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('delta')
    plt.title(f"Evaluation of Training Iteration {ii}")
    plt.show()

    fig, ax = plt.subplots()
    for i in range(policy.v_filt.cbf.rs.shape[0]):
        circ = plt.Circle((policy.v_filt.cbf.cs[i, 0].item(), policy.v_filt.cbf.cs[i, 1].item()),
                          policy.v_filt.cbf.rs[i].item())
        ax.add_patch(circ)
    ax.plot(x[::20, :, 0].cpu().numpy().T, x[::20, :, 1].cpu().numpy().T)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    plt.title(f"Evaluation of Training Iteration {ii}")
    plt.show()