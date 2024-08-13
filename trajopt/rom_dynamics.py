import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
from deep_tube_learning.utils import yaw2rot
import matplotlib.cm as cm
import torch
from isaacgym.torch_utils import *


class RomDynamics(ABC):
    """
    Abstract class for Reduced order Model Dynamics
    """
    n: int  # Dimension of state
    m: int  # Dimension of input

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi', device='cuda'):
        """
        Common constructor functionality
        :param dt: time discretization
        :param z_min: lower state bound
        :param z_max: upper state bound
        :param v_min: lower input bound
        :param v_max: upper input bound
        :param backend: 'casadi' for when using dynamics for a casadi optimization program,
               'numpy' for use with numpy arrays
        """
        self.dt = dt
        self.v_min = v_min
        self.v_max = v_max
        self.z_min = z_min
        self.z_max = z_max
        self.n_robots = n_robots
        self.vel_inds = None
        self.device = device

        if backend == 'casadi':
            self.zero_mat = lambda r, c: ca.MX(r, c)
            self.zero_vec = lambda n: ca.MX(n, 1)
            self.const_mat = lambda m: ca.DM(m)
            self.sin = lambda x: ca.sin(x)
            self.cos = lambda x: ca.cos(x)
            self.stack = lambda lst: ca.horzcat(*lst)
            self.vstack = lambda lst: ca.vertcat(*lst)
            self.arctan = lambda y, x: ca.arctan2(y, x)

        elif backend == 'numpy':
            self.zero_mat = lambda r, c: np.zeros((r, c))
            self.zero_vec = lambda n: np.zeros((n,))
            self.const_mat = lambda m: np.array(m)
            self.sin = lambda x: np.sin(x)
            self.cos = lambda x: np.cos(x)
            self.stack = lambda lst: np.hstack(lst)
            self.vstack = lambda lst: np.vstack(lst)
            self.arctan2 = lambda y, x: np.arctan2(y, x)
            self.maximum = np.maximum
            self.minimum = np.minimum
            self.repeat = lambda arr, n, ax: np.repeat(arr, n, axis=ax)
            self.squeeze = torch.squeeze
        elif backend == 'torch':
            self.zero_mat = lambda r, c: torch.zeros((r, c), device=device)
            self.zero_vec = lambda n: torch.zeros((n,), device=device)
            self.const_mat = lambda m: torch.tensor(m, device=device)
            self.sin = lambda x: torch.sin(x)
            self.cos = lambda x: torch.cos(x)
            self.stack = lambda lst: torch.hstack(lst)
            self.vstack = lambda lst: torch.vstack(lst)
            self.arctan2 = lambda y, x: torch.arctan2(y, x)
            self.maximum = torch.max
            self.minimum = torch.min
            self.repeat = lambda arr, n, ax: torch.repeat_interleave(arr, n, dim=ax)
            self.squeeze = torch.squeeze

    @abstractmethod
    def f(self, z, v):
        """
        Dynamics function
        :param z: current state
        :param v: input
        :return: next state
        """
        raise NotImplementedError

    @abstractmethod
    def proj_z(self, x):
        """
        Projects the full order robot's CoM state (p, q, v, w) in R^13 onto the RoM state
        :param x: the robot's CoM state
        :return: projection of the full order state onto the CoM state
        """
        raise NotImplementedError

    @abstractmethod
    def des_pose_vel(self, z, v):
        """
        Computes the desired pose (x,y,yaw) and velocity (xdot, ydot, yawdot) given the RoM state and input
        :param z: RoM state
        :param v: RoM input
        :return: the desired pose and velocity of the robot
        """
        raise NotImplementedError

    def clip_v(self, v):
        return self.maximum(self.minimum(v, self.v_max), self.v_min)

    def compute_state_dependent_input_bounds(self, z):
        return self.repeat(self.v_min[None, :], z.shape[0], 0), self.repeat(self.v_max[None, :], z.shape[0], 0)

    @abstractmethod
    def clip_v_z(self, z, v):
        """
        Clips the input, v, to a valid input which respects input bounds, and when the dyamics are applied,
        will not result in velocities which violate the state bounds (when applicable)
        :param z: state
        :param v: input
        :return: clipped input which is valid in the given state
        """
        raise NotImplementedError

    @staticmethod
    def plot_spacial(ax, xt, c=None):
        """
        Plots the x, y spatial trajectory on the given axes with a color gradient to indicate time series.
        :param ax: axes on which to plot
        :param xt: state trajectory
        :param c: color/line type
        """
        N = xt.shape[0]
        colors = cm.viridis(np.linspace(0, 1, N))  # Use the 'viridis' colormap

        # Plot segments with color gradient
        if c is None:
            for i in range(N - 1):
                ax.plot(xt[i:i + 2, 0], xt[i:i + 2, 1], color=colors[i])
            scatter = ax.scatter(xt[:, 0], xt[:, 1], c=np.linspace(0, 1, N), cmap='viridis', s=10,
                                 edgecolor='none')  # Plot points for better visibility

            # Add color bar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Time')
        else:
            ax.plot(xt[:, 0], xt[:, 1], c)

    @staticmethod
    def plot_tube(ax, xt, wt, c='g'):
        """
        Plots the tube on given axes
        :param ax: axes on which to plot
        :param xt: state trajectory
        :param wt: tube width
        :param c: color/line type
        """
        for i in range(xt.shape[0]):
            # TODO: vector tube plotting
            if wt.shape[1] == 1:
                xc = xt[i, 0]
                yc = xt[i, 1]
                circ = plt.Circle((xc, yc), wt[i], color=c, fill=False)
                ax.add_patch(circ)

    def plot_ts(self, axs, xt, ut):
        """
        Plots states and inputs over time
        :param axs: size 2 array of axes on which to plot states (first) and inputs (second)
        :param xt: state trajectory
        :param ut: input trajectory
        """
        N = xt.shape[0]
        ts = np.linspace(0, N * self.dt, N)
        axs[0].plot(ts, xt)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('State')

        axs[1].plot(ts[:-1], ut)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Input')

    def get_weighting_vector(self, reward_weighting):
        '''to generate the reward weighting used in trajectory tracking in legged_robot'''
        raise NotImplementedError

class SingleInt2D(RomDynamics):
    n = 2   # [x, y]
    m = 2   # [vx, vy]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi', device='cuda'):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend, device=device)
        self.A = self.const_mat([[1.0, 0], [0, 1.0]])
        self.B = self.const_mat([[dt, 0], [0, dt]])
        self.vel_inds = self.const_mat([False, False])

    def f(self, x, u):
        return (self.A @ x.T).T + (self.B @ u.T).T

    def proj_z(self, x):
        return x[..., :2]

    def des_pose_vel(self, z, v):
        return self.stack((z, self.arctan2(v[:, 1], v[:, 0])[:, None])), self.stack((v, self.zero_mat(v.shape[0], 1)))

    def clip_v_z(self, z, v):
        return v

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y'])
        axs[1].legend(['vx', 'vy'])

    def get_weighting_vector(self, reward_weighting):
        return torch.tensor([reward_weighting.position, reward_weighting.position], dtype=torch.float32,
                            device=self.device)


class DoubleInt2D(RomDynamics):
    n = 4   # [x, y, vx, vy]
    m = 2   # [ax, ay]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi', device='cuda'):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend, device=device)
        self.A = self.const_mat([[1.0, 0, dt, 0], [0, 1.0, 0, dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
        self.B = self.const_mat([[0, 0], [0, 0], [dt, 0], [0, dt]])
        self.vel_inds = self.const_mat([False, False, True, True])

    def f(self, x, u):
        return (self.A @ x.T).T + (self.B @ u.T).T

    def proj_z(self, x):
        return self.stack((x[..., :2], x[..., 7:9]))

    def des_pose_vel(self, z, v):
        return (self.stack((z[:, :2], self.arctan2(z[:, 3], z[:, 2])[:, None])),
                self.stack((z[:, 2:], self.zero_mat(v.shape[0], 1))))

    def compute_state_dependent_input_bounds(self, z):
        """
        Because there are state bounds on the velocity, in some states we cannot use the full input range.
        For instance, if the x velocity is at its maximum, applying a positive x acceleration
        will move the state outside the state bounds.
        This function computes new input bounds which ensure that inputs within these bounds will not result in state
        violation (for the velocity state bounds)
        :param z: current state
        :return: state-dependent input bounds (lower, upper)
        """
        v_max_z = self.minimum(self.v_max, (self.z_max[2:] - z[:, 2:]) / self.dt)
        v_min_z = self.maximum(self.v_min, (self.z_min[2:] - z[:, 2:]) / self.dt)
        return v_min_z, v_max_z

    def clip_v_z(self, z, v):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        return self.maximum(self.minimum(v, v_max_z), v_min_z)

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'vx', 'vy'])
        axs[1].legend(['ax', 'ay'])

    def get_weighting_vector(self, reward_weighting):
        return torch.tensor([reward_weighting.position, reward_weighting.position,
                             reward_weighting.velocity, reward_weighting.velocity], dtype=torch.float32,
                            device=self.device)


class Unicycle(RomDynamics):
    n = 3   # [x, y, theta]
    m = 2   # [v, omega]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi', device='cuda'):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend, device=device)
        self.vel_inds = self.const_mat([False, False, False])

    def f(self, x, u):
        gu = self.zero_mat(self.n_robots, self.n)
        gu[:, 0] = u[:, 0] * self.cos(x[:, 2])
        gu[:, 1] = u[:, 0] * self.sin(x[:, 2])
        gu[:, 2] = u[:, 1]
        return x + self.dt * gu

    def proj_z(self, x):
        quat = x[:, 3:7]
        rot = Rotation.from_quat(quat)
        eul = rot.as_euler('xyz', degrees=False)
        return self.stack((x[..., :2], eul[..., -1][:, None]))

    def des_pose_vel(self, z, v):
        vx = v[:, 0] * self.cos(z[:, 2])
        vy = v[:, 0] * self.sin(z[:, 2])
        om = v[:, 1]
        return z[:, :3], self.stack((vx[:, None], vy[:, None], om[:, None]))

    def clip_v_z(self, z, v):
        return v

    @staticmethod
    def plot_spacial(ax, xt, c='-b'):
        RomDynamics.plot_spacial(ax, xt, c=c)
        ax.quiver(xt[:, 0], xt[:, 1], np.cos(xt[:, 2]), np.sin(xt[:, 2]))

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta'])
        axs[1].legend(['v', 'omega'])

    def get_weighting_vector(self, reward_weighting):
        return torch.tensor([reward_weighting.position, reward_weighting.position,
                             reward_weighting.orientation], dtype=torch.float32, device=self.device)

class LateralUnicycle(Unicycle):
    n = 3   # [x, y, theta]
    m = 3   # [v, v_perp, omega]

    def f(self, x, u):
        gu = self.zero_mat(self.n_robots, self.n)
        gu[:, 0] = u[:, 0] * self.cos(x[:, 2]) - u[:, 1] * self.sin(x[:, 2])
        gu[:, 1] = u[:, 0] * self.sin(x[:, 2]) + u[:, 1] * self.cos(x[:, 2])
        gu[:, 2] = u[:, 2]
        return x + self.dt * gu

    def des_pose_vel(self, z, v):
        vx = v[:, 0] * self.cos(z[:, 2]) - v[:, 1] * self.sin(z[:, 2])
        vy = v[:, 0] * self.sin(z[:, 2]) + v[:, 1] * self.cos(z[:, 2])
        om = v[:, 1]
        return z[:, :3], self.stack((vx[:, None], vy[:, None], om[:, None]))

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta'])
        axs[1].legend(['v', 'v_perp', 'omega'])

    def get_weighting_vector(self, reward_weighting):
        return torch.tensor([reward_weighting.position, reward_weighting.position,
                             reward_weighting.orientation, reward_weighting.velocity,
                             reward_weighting.velocity, reward_weighting.angular_velocity], dtype=torch.float32,
                            device=self.device)


class ExtendedUnicycle(Unicycle):
    n = 5   # [x, y, theta, v, omega]
    m = 2   # [a, alpha]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi', device='cuda'):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend, device=device)
        self.vel_inds = self.const_mat([False, False, False, True, True])

    def f(self, x, u):
        gu = self.zero_mat(self.n_robots, self.n)
        gu[:, 0] = x[:, 3] * self.cos(x[:, 2])
        gu[:, 1] = x[:, 3] * self.sin(x[:, 2])
        gu[:, 2] = x[:, 4]
        gu[:, 3] = u[:, 0]
        gu[:, 4] = u[:, 1]
        return x + self.dt * gu

    def des_pose_vel(self, z, v):
        vx = z[:, 3] * self.cos(z[:, 2])
        vy = z[:, 3] * self.sin(z[:, 2])
        om = z[:, 4]
        return z[:, :3], self.stack((vx[:, None], vy[:, None], om[:, None]))

    def proj_z(self, x):
        quat = x[:, 3:7]
        v = x[:, 7:9]
        rot = Rotation.from_quat(quat)
        eul = rot.as_euler('xyz', degrees=False)
        v_local = np.squeeze(yaw2rot(eul[..., -1]) @ v[:, :, None])
        return self.stack((x[..., :2], eul[..., -1][:, None], v_local[:, 0][:, None], x[:, -1][:, None]))

    def compute_state_dependent_input_bounds(self, z):
        """
        Because there are state bounds on the velocity, in some states we cannot use the full input range.
        For instance, if the x velocity is at its maximum, applying a positive x acceleration
        will move the state outside the state bounds.
        This function computes new input bounds which ensure that inputs within these bounds will not result in state
        violation (for the velocity state bounds)
        :param z: current state
        :return: state-dependent input bounds (lower, upper)
        """
        v_max_z = self.minimum(self.v_max, (self.z_max[3:] - z[:, 3:]) / self.dt)
        v_min_z = self.maximum(self.v_min, (self.z_min[3:] - z[:, 3:]) / self.dt)
        return v_min_z, v_max_z

    def clip_v_z(self, z, v):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        return self.maximum(self.minimum(v, v_max_z), v_min_z)

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta', 'v', 'omega'])
        axs[1].legend(['a', 'alpha'])

    def get_weighting_vector(self, reward_weighting):
        return torch.tensor([reward_weighting.position, reward_weighting.position,
                             reward_weighting.orientation, reward_weighting.velocity,
                             reward_weighting.angular_velocity], dtype=torch.float32,
                            device=self.device)


class ExtendedLateralUnicycle(ExtendedUnicycle):
    n = 6   # [x, y, theta, v, v_perp, omega]
    m = 3   # [a, a_perp, alpha]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi', device='cuda'):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend, device=device)
        self.vel_inds = self.const_mat([False, False, False, True, True, True])

    def f(self, x, u):
        gu = self.zero_mat(self.n_robots, self.n)
        gu[:, 0] = x[:, 3] * self.cos(x[:, 2]) - x[:, 4] * self.sin(x[:, 2])
        gu[:, 1] = x[:, 3] * self.sin(x[:, 2]) + x[:, 4] * self.cos(x[:, 2])
        gu[:, 2] = x[:, 5]
        gu[:, 3] = u[:, 0]
        gu[:, 4] = u[:, 1]
        gu[:, 5] = u[:, 2]
        return x + self.dt * gu

    def des_pose_vel(self, z, v):
        vx = z[:, 3] * self.cos(z[:, 2]) - z[:, 4] * self.sin(z[:, 2])
        vy = z[:, 3] * self.sin(z[:, 2]) + z[:, 4] * self.cos(z[:, 2])
        om = z[:, 5]
        return z[:, :3], self.stack((vx[:, None], vy[:, None], om[:, None]))

    def proj_z(self, x):
        quat = x[:, 3:7]
        v = x[:, 7:9]
        rot = Rotation.from_quat(quat)
        eul = rot.as_euler('xyz', degrees=False)
        v_local = self.squeeze(yaw2rot(eul[..., -1]) @ v[:, :, None])
        return self.stack((x[..., :2], eul[..., -1][:, None], v_local, x[:, -1][:, None]))

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta', 'v', 'v_perp', 'omega'])
        axs[1].legend(['a', 'a_perp', 'alpha'])

    def get_weighting_vector(self, reward_weighting):
        return torch.tensor([reward_weighting.position, reward_weighting.position,
                             reward_weighting.orientation, reward_weighting.velocity,
                             reward_weighting.velocity, reward_weighting.angular_velocity], dtype=torch.float32,
                            device=self.device)

class TrajectoryGenerator:

    def __init__(self, rom, t_sampler, weight_sampler, N=4, freq_low=0.01, freq_high=10, seed=42,
                 backend='numpy', device='cuda', prob_stationary=.01, DN=1):
        self.rom = rom
        self.device = device
        if backend == 'numpy':
            self.zeros = np.zeros
            self.ones = lambda sh, dtype: np.ones(sh, dtype=dtype)
            self.ones_like = lambda sh, dt: np.ones_like(dt, dtype=dt)
            self.zeros_like = np.zeros_like
            self.rng = np.random.default_rng(seed)
            self.arange = np.arange
            self.concatenate = lambda d, dim: np.concatenate(d, axis=dim)
            self.choice = lambda h, sh: np.random.choice(np.arrange(h), size=(*sh, 1))
            self.nonzero = lambda n: n.nonzero()[0]
            self.sin = np.sin
            self.sum = np.sum
            self.array = np.array
            self.pi = np.pi

            def uniform(low, high, size):
                return self.rng.uniform(low, high, size)
        elif backend == 'torch':
            self.zeros = lambda sh: torch.zeros(sh, device=device)
            self.ones = lambda sh, dtype: torch.ones(sh, dtype=dtype, device=device)
            self.ones_like = lambda sh, dt: torch.ones_like(dt, dtype=dt, device=device)
            self.zeros_like = lambda m: torch.zeros_like(m, device=device)
            self.arange = lambda l: torch.arange(l, device=device)
            self.concatenate = lambda d, dim: torch.concatenate(d, dim=dim)
            self.choice = lambda h, sh: torch.randint(0, h, (*sh, 1), device=device)
            self.nonzero = lambda n: torch.nonzero(n)
            self.sin = torch.sin
            self.sum = torch.sum
            self.array = lambda a: torch.tensor(a, device=device)
            self.pi = torch.pi

            def uniform(low, high, size):
                return (high - low) * torch.rand(*size, device=device) + low

        else:
            raise ValueError(f'Unsupported backend: {device}')
        self.uniform = uniform
        self.N = N
        self.DN = DN
        self.weights = self.zeros((self.rom.n_robots, 4))
        self.t_final = self.zeros((self.rom.n_robots,))
        self.t = self.zeros((self.rom.n_robots,))
        self.sample_hold_input = self.zeros((self.rom.n_robots, self.rom.m))
        self.extreme_input = self.zeros((self.rom.n_robots, self.rom.m))
        self.ramp_t_start = self.zeros((self.rom.n_robots,))
        self.ramp_v_start = self.zeros((self.rom.n_robots, self.rom.m))
        self.ramp_v_end = self.uniform(self.rom.v_min, self.rom.v_max, size=(self.rom.n_robots, self.rom.m))
        self.sin_mag = self.zeros((self.rom.n_robots, self.rom.m))
        self.sin_freq = self.zeros((self.rom.n_robots, self.rom.m))
        self.sin_off = self.zeros((self.rom.n_robots, self.rom.m))
        self.sin_mean = self.zeros((self.rom.n_robots, self.rom.m))
        self.t_sampler = t_sampler
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.weight_sampler = weight_sampler
        self.trajectory = self.zeros((self.rom.n_robots, self.N // self.DN, self.rom.n))
        self.v = self.zeros((self.rom.n_robots, self.rom.m))
        self.prob_stationary = prob_stationary
        self.stationary_inds = self.zeros((self.rom.n_robots,)).bool()

    def reset_inputs(self):
        t_mask = self.ones_like(self.t_final, bool)
        z = self.zeros((self.rom.n_robots, self.rom.n))
        self.t_final = self.zeros((self.rom.n_robots,))
        self.resample(t_mask, z)

    def resample(self, idx, z):
        if len(idx) > 0:
            v_min, v_max = self.rom.compute_state_dependent_input_bounds(z[idx, :])
            self._resample_const_input(idx, v_min, v_max)
            self._resample_ramp_input(idx, z, v_min, v_max)
            self._resample_extreme_input(idx, v_min, v_max)
            self._resample_sinusoid_input(idx, v_min, v_max)
            self._resample_t_final(idx)
            self._resample_weight(idx)
            n = self.sum(idx) if idx.dtype == bool else len(idx)
            self.stationary_inds[idx] = self.uniform(self.array([0.0]), self.array([1.0]), (n, 1)).squeeze() < self.prob_stationary

    def _resample_t_final(self, idx):
        self.t_final[idx] += self.t_sampler.sample(len(idx))

    def _resample_weight(self, idx):
        self.weights[idx, :] = self.weight_sampler.sample(len(idx))

    def _resample_const_input(self, idx, v_min, v_max):
        self.sample_hold_input[idx, :] = self.uniform(v_min, v_max, size=(len(idx), self.rom.m))

    def _resample_ramp_input(self, idx, z, v_min, v_max):
        self.ramp_v_start[idx, :] = self.rom.clip_v_z(z[idx, :], self.ramp_v_end[idx, :])
        self.ramp_v_end[idx, :] = self.uniform(v_min, v_max, size=(len(idx), self.rom.m))
        self.ramp_t_start[idx] = self.t_final[idx]

    def _resample_extreme_input(self, t_mask,v_min, v_max):
        arr = self.concatenate((v_min[:, :, None], self.zeros_like(v_min)[:, :, None], v_max[:, :, None]), -1)
        mask = self.arange(3)[None, None, :] == self.choice(3, v_min.shape)
        self.extreme_input[t_mask, :] = arr[mask].reshape(v_min.shape)

    def _resample_sinusoid_input(self, idx, v_min, v_max):
        self.sin_mag[idx, :] = self.uniform(self.zeros_like(v_max), (v_max - v_min) / 2, size=(len(idx), self.rom.m))
        self.sin_mean[idx, :] = self.uniform(v_min + self.sin_mag[idx, :], v_max - self.sin_mag[idx, :], size=(len(idx), self.rom.m))
        self.sin_freq[idx, :] = self.uniform(self.array([self.freq_low]), self.array([self.freq_high]), size=(len(idx), self.rom.m))
        self.sin_off[idx, :] = self.uniform(self.array([-self.pi]), self.array([-self.pi]), size=(len(idx), self.rom.m))

    def _const_input(self):
        return self.sample_hold_input

    def _ramp_input_t(self, t):
        return self.ramp_v_start + \
            (self.ramp_v_end - self.ramp_v_start) * ((t - self.ramp_t_start) / (self.t_final - self.ramp_t_start))[:, None]

    def _extreme_input(self):
        return self.extreme_input

    def _sinusoid_input_t(self, t):
        return self.sin_mag * self.sin(self.sin_freq * t[:, None] + self.sin_off) + self.sin_mean

    def get_input_t(self, t, z):
        idx = self.nonzero(t > self.t_final).reshape((-1,))
        self.resample(idx, z)
        return self.weights[:, 0][:, None] * self.rom.clip_v_z(z, self._const_input()) + \
            self.weights[:, 1][:, None] * self.rom.clip_v_z(z, self._ramp_input_t(t)) + \
            self.weights[:, 2][:, None] * self.rom.clip_v_z(z, self._extreme_input()) + \
            self.weights[:, 3][:, None] * self.rom.clip_v_z(z, self._sinusoid_input_t(t))

    def step(self):
        # Get input to apply for trajectory
        self.v = self.get_input_t(self.t, self.trajectory[:, -1, :])
        self.v[self.stationary_inds, :] = 0
        z_next = self.rom.f(self.trajectory[:, -1, :], self.v)
        mask = self.stationary_inds[:, None] & self.rom.vel_inds
        z_next[mask] = 0
        self.trajectory[:, :-1, :] = self.trajectory[:, 1:, :]
        self.trajectory[:, -1, :] = z_next
        self.t += self.rom.dt

    def step_idx(self, idx):
        # Get input to apply for trajectory
        self.v = self.get_input_t(self.t, self.trajectory[:, -1, :])
        self.v[self.stationary_inds, :] = 0
        z_next = self.rom.f(self.trajectory[idx, -1, :], self.v[idx, :])
        mask = self.stationary_inds[:, None] & self.rom.vel_inds
        z_next[mask[idx, :]] = 0
        self.trajectory[idx, :-1, :] = self.trajectory[idx, 1:, :]
        self.trajectory[idx, -1, :] = z_next
        self.t[idx] += self.rom.dt

    def reset(self, z):
        self.reset_idx(self.ones((self.rom.n_robots,), bool), z)

    def reset_idx(self, idx, z):
        self.trajectory[idx, :, :] = self.zeros((len(idx), self.N // self.DN, self.rom.n))
        self.trajectory[idx, -1, :] = z[idx, :]
        self.t[idx] = 0
        self.t_final[idx] = 0
        self.resample(idx, z)

        for t in range(self.N - 1):
            self.step_idx(idx)


class ZeroTrajectoryGenerator(TrajectoryGenerator):

    def resample(self, idx, z):
        self.stationary_inds[idx] = True

    def get_input_t(self, t, z):
        return self.zeros((self.rom.n_robots, self.rom.m))


class SquareTrajectoryGenerator(TrajectoryGenerator):

    def resample(self, idx, z):
        pass

    def get_input_t(self, t, z):
        v = self.zeros((self.rom.n_robots, self.rom.m))
        if isinstance(self.rom, SingleInt2D):
            c1 = 2 / self.rom.v_max[1]
            c2 = c1 + 1 / self.rom.v_max[0]
            c3 = c2 + 2 / abs(self.rom.v_min[1])
            c4 = c3 + 1 / abs(self.rom.v_min[0])
            v[(0 <= t) & (t < c1), 1] = self.rom.v_max[1] / 2
            v[(c1 <= t) & (t < c2), 0] = self.rom.v_max[0]
            v[(c2 <= t) & (t < c3), 1] = self.rom.v_min[1] / 2
            v[(c3 <= t) & (t < c4), 0] = self.rom.v_min[1]
        elif isinstance(self.rom, DoubleInt2D):
            c0 = self.rom.z_max[3] / 2 / self.rom.v_max[1]
            c1 = c0 + (1 - 2 * (0.5 * self.rom.v_max[1] * c0**2)) / (self.rom.z_max[3] / 2)
            c2 = c1 + self.rom.z_min[3] / 2 / self.rom.v_min[1]
            c3 = c2
            c4 = c3 + self.rom.z_max[2] / self.rom.v_max[0]
            c5 = c4 + (1 - 2 * (0.5 * self.rom.v_max[0] * (c4 - c3) ** 2)) / (self.rom.z_max[2] / 2)
            c6 = c5 + self.rom.z_min[2] / self.rom.v_min[0]
            c7 = c6
            c8 = c7 + self.rom.z_min[3] / 2 / self.rom.v_min[1]
            c9 = c8 + (1 - 2 * (0.5 * abs(self.rom.v_min[1]) * (c8 - c7) ** 2)) / (abs(self.rom.z_min[3]) / 2)
            c10 = c9 + self.rom.z_max[3] / 2 / self.rom.v_max[1]
            c11 = c10
            c12 = c11 + self.rom.z_min[2] / self.rom.v_min[0]
            c13 = c12 + (1 - 2 * (0.5 * abs(self.rom.v_min[0]) * (c12 - c11) ** 2)) / (abs(self.rom.z_min[2]) / 2)
            c14 = c13 + self.rom.z_max[2] / self.rom.v_max[0]

            v[(0 <= t) & (t < c0), 1] = self.rom.v_max[1].float()
            v[(c1 <= t) & (t < c2), 1] = self.rom.v_min[1].float()
            v[(c3 <= t) & (t < c4), 0] = self.rom.v_max[0].float()
            v[(c5 <= t) & (t < c6), 0] = self.rom.v_min[0].float()
            v[(c7 <= t) & (t < c8), 1] = self.rom.v_min[1].float()
            v[(c9 <= t) & (t < c10), 1] = self.rom.v_max[1].float()
            v[(c11 <= t) & (t < c12), 0] = self.rom.v_min[0].float()
            v[(c13 <= t) & (t < c14), 0] = self.rom.v_max[0].float()
        else:
            raise ValueError("Only SingleInt2D and DoubleInt2D are supported")
        return v

    def reset_idx(self, idx, z):
        z[:, self.rom.vel_inds] = 0
        super().reset_idx(idx, z)


class CircleTrajectoryGenerator(TrajectoryGenerator):

    def resample(self, idx, z):
        self.center = torch.clone(z.detach())[:, :2]
        self.center[:, 0] -= 0.5

    def get_input_t(self, t, z):
        v = self.zeros((self.rom.n_robots, self.rom.m))
        if isinstance(self.rom, SingleInt2D):
            e = z - self.center
            v[:, 0] = - e[:, 1]
            v[:, 1] = e[:, 0]
            v += -(e - 0.5 * e / torch.linalg.norm(v, dim=-1, keepdim=True))
            v = v / torch.linalg.norm(v, dim=-1, keepdim=True) * torch.min(torch.minimum(self.rom.v_max, torch.abs(self.rom.v_min)))
        elif isinstance(self.rom, DoubleInt2D):
            m = torch.min(torch.minimum(self.rom.v_max, torch.abs(self.rom.v_min)))
            z_des = self.center + 0.5 * torch.concatenate([torch.cos(t / m)[:, None], torch.sin(t / m)[:, None]], dim=-1)
            v_des = 0.5 * torch.concatenate([-torch.sin(t / m)[:, None], torch.cos(t / m)[:, None]], dim=-1) / m
            v = self.rom.clip_v_z(z, -4 * (z[:, :2] - z_des) - 4 * (z[:, 2:] - v_des))
        else:
            raise ValueError("Only SingleInt2D and DoubleInt2D are supported")
        return v

