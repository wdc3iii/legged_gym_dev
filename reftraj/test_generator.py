import matplotlib.pyplot as plt
import numpy as np
from ref_generator import ReferenceTrajectoryGenerator
import torch

def test_trajectory_generator():
    n_robots = 3
    horizon_length = 50
    dt = 0.1
    dof = 6  # Assuming each robot has 6 degrees of freedom
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create an instance of the trajectory generator
    generator = ReferenceTrajectoryGenerator(n_robots, horizon_length, dt, dof, device)

    # Reset the generator
    generator.reset()

    # Generate trajectories
    trajectories = []
    for _ in range(horizon_length):
        trajectory = generator.step()
        trajectories.append(trajectory.clone())

    # Convert list of trajectories to torch tensor for easier handling
    trajectories = torch.stack(trajectories, dim=0).cpu().numpy()

    # Plot the trajectories for each joint of the first robot
    fig, axs = plt.subplots(dof, 1, figsize=(10, 15))
    for j in range(dof):
        for i in range(n_robots):
            axs[j].plot(trajectories[:, i, :, 0, j].flatten(), label=f'Robot {i+1} Position')
            axs[j].plot(trajectories[:, i, :, 1, j].flatten(), label=f'Robot {i+1} Velocity', linestyle='--')
        axs[j].set_xlabel('Time step')
        axs[j].set_ylabel(f'Joint {j+1} position and velocity')
        axs[j].legend()
    plt.show()

if __name__ == "__main__":
    test_trajectory_generator()