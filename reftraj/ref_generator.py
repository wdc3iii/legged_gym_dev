import torch


class ReferenceTrajectoryGenerator:
    """
    Class to generate reference trajectories for multiple robots.
    """

    def __init__(self, n_robots, horizon_length, dt, dof, device='cpu'):
        """
        Initialize the generator.

        :param n_robots: Number of robots
        :param horizon_length: Length of the trajectory horizon
        :param dt: Time step
        :param dof: Degrees of freedom per robot
        :param device: Device to use ('cpu' or 'cuda')
        """
        self.n_robots = n_robots
        self.horizon_length = horizon_length
        self.dt = dt
        self.dof = dof
        self.device = device
        self.current_time = 0
        # Initialize trajectory with zeros: [num_robots, horizon_length, 2 (pos and vel), dof]
        self.trajectory = torch.zeros((n_robots, horizon_length, 2, dof), device=self.device)
        self.initialize_reference_trajectory()

    def initialize_reference_trajectory(self):
        """
        Initialize the reference trajectory with specific values for positions and zeros for velocities.
        """
        joint_positions = [0, 0.4, -0.8, 0, -0.4, 0.8, 0, 0.4, -0.8, 0, -0.4, 0.8]
        pos_length = len(joint_positions)
        repeated_positions = joint_positions * (self.dof // pos_length) + joint_positions[:self.dof % pos_length]
        repeated_positions = torch.tensor(repeated_positions, device=self.device, dtype=torch.float)

        for i in range(self.n_robots):
            for t in range(self.horizon_length):
                self.trajectory[i, t, 0, :] = repeated_positions  # Set positions
                self.trajectory[i, t, 1, :] = 0  # Velocities are zero

    def step(self):
        """
        Generate the next step in the trajectory.
        """
        self.current_time += self.dt
        return self.trajectory

    def reset(self):
        """
        Reset the generator.
        """
        self.current_time = 0
        self.initialize_reference_trajectory()