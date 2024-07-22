import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from ref_generator import ReferenceTrajectoryGenerator

def test_trajectory_generator():
    n_robots = 3
    dt = 0.1
    dof = 14  # Updated DoF
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    csv_directory = 'adam_reference_trajectories/processed'  # Replace with your directory path

    # Create an instance of the trajectory generator
    generator = ReferenceTrajectoryGenerator(n_robots, dt, dof, csv_directory, device)

    # Reset the generator
    generator.reset()

    # Simulate different speeds and generate trajectories
    num_steps = 100
    speeds = torch.tensor([0.0, 0.3, 0.7, 1.2, 1.0, 0.5], device=device).repeat(n_robots // 2 + 1)[:n_robots]
    trajectories = []

    # Create random speed changes
    speed_changes = np.random.choice([True, False], size=num_steps, p=[0.1, 0.9])

    for step in range(num_steps):  # 10 seconds of simulation
        if speed_changes[step]:
            speeds = torch.rand(n_robots, device=device) * 1.5  # Random speeds between 0 and 1.5
        trajectory = generator.step(speeds)
        trajectories.append(trajectory.clone())

    # Convert list of trajectories to torch tensor for easier handling
    trajectories = torch.stack(trajectories, dim=0).cpu().numpy()

    # Plot the trajectories for each joint of the first robot
    fig, axs = plt.subplots(dof, 2, figsize=(20, 30))
    for j in range(dof):
        for i in range(n_robots):
            axs[j, 0].plot(trajectories[:, i, 0, j].flatten(), label=f'Robot {i+1} Position')
            axs[j, 1].plot(trajectories[:, i, 1, j].flatten(), label=f'Robot {i+1} Velocity', linestyle='--')
        axs[j, 0].set_xlabel('Time step')
        axs[j, 0].set_ylabel(f'Joint {j+1} Position')
        axs[j, 1].set_xlabel('Time step')
        axs[j, 1].set_ylabel(f'Joint {j+1} Velocity')
        axs[j, 0].legend()
        axs[j, 1].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_trajectory_generator()
