import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

class ReferenceTrajectoryGenerator:
    """
    Class to generate reference trajectories for multiple robots.
    """

    def __init__(self, n_robots, dt, dof, csv_directory, device='cpu'):
        """
        Initialize the generator.

        :param n_robots: Number of robots
        :param dt: Time step
        :param dof: Degrees of freedom per robot
        :param csv_directory: Directory containing the reference trajectory CSV files
        :param device: Device to use ('cpu' or 'cuda')
        """
        self.n_robots = n_robots
        self.dt = dt
        self.dof = dof
        self.device = device
        self.csv_directory = csv_directory
        self.current_times = torch.zeros(n_robots, device=self.device)
        self.current_speeds = torch.full((n_robots,), -1.0, device=self.device)
        self.current_trajectories = [None] * n_robots
        self.current_trajectory_times = [None] * n_robots

        self.speed_bins = torch.arange(0, 1.6, 0.1, device=self.device)
        self.initialize_reference_trajectory()

    def initialize_reference_trajectory(self):
        """
        Initialize the reference trajectory with zeros.
        """
        self.trajectory = torch.zeros((self.n_robots, 2, self.dof), device=self.device)

    def load_trajectory_from_csv(self, robot_index, speed_bin):
        """
        Load the reference trajectory from the CSV file corresponding to the given speed bin.

        :param robot_index: Index of the robot.
        :param speed_bin: The binned speed value.
        """
        file_name = f"joint_data_speed_{speed_bin:.1f}.csv"
        csv_path = os.path.join(self.csv_directory, file_name)

        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            times = data['Time'].unique()
            positions = []
            velocities = []
            joints = data['Joint'].unique()
            for joint in joints:
                joint_data = data[data['Joint'] == joint]
                positions.append(joint_data['Position'].values)
                velocities.append(joint_data['Velocity'].values)

            positions = torch.tensor(positions, device=self.device, dtype=torch.float).T
            velocities = torch.tensor(velocities, device=self.device, dtype=torch.float).T

            self.current_trajectories[robot_index] = torch.stack((positions, velocities),
                                                                 dim=1)  # Shape: (timesteps, 2, dof)
            self.current_trajectory_times[robot_index] = torch.tensor(times, device=self.device, dtype=torch.float)
        else:
            print(f"CSV file for speed {speed_bin:.1f} not found. Using zeros.")

    def step(self, speeds):
        """
        Generate the next step in the trajectory based on the current speeds of the robots.

        :param speeds: The current speeds of the robots.
        """
        for i in range(self.n_robots):
            # Bin the speed
            closest_speed_bin = self.speed_bins[torch.argmin(torch.abs(self.speed_bins - speeds[i]))].item()

            if closest_speed_bin != self.current_speeds[i]:
                self.current_speeds[i] = closest_speed_bin
                self.load_trajectory_from_csv(i, closest_speed_bin)
                self.current_times[i] = 0

            if self.current_trajectories[i] is not None:
                # Find the closest time index in the current_trajectory_times array
                time_diff = torch.abs(self.current_trajectory_times[i] - self.current_times[i])
                closest_time_index = torch.argmin(time_diff).item()

                self.trajectory[i, :, :] = self.current_trajectories[i][closest_time_index, :, :]

                # Check if we have reached the end of the trajectory and loop around if necessary
                if self.current_times[i] >= self.current_trajectory_times[i][-1]:
                    self.current_times[i] = 0

            self.current_times[i] += self.dt

        return self.trajectory

    def reset(self):
        """
        Reset the generator.
        """
        self.current_times = torch.zeros(self.n_robots, device=self.device)
        self.current_speeds = torch.full((self.n_robots,), -1.0, device=self.device)
        self.current_trajectories = [None] * self.n_robots
        self.current_trajectory_times = [None] * self.n_robots
        self.initialize_reference_trajectory()
