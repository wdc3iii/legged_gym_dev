import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
from deep_tube_learning.utils import yaw2rot


class RomDynamics(ABC):
    """
    Abstract class for Reduced order Model Dynamics
    """
    n: int  # Dimension of state
    m: int  # Dimension of input

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi'):
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

    @abstractmethod
    def sample_uniform_bounded_v(self, z):
        """
        Samples an input, v which respects input bounds, and, when the dynamics are applied,
        will not result in velocities which violate the state bounds (when applicable)
        :param z: current state
        :return: valid input in the given state
        """
        raise NotImplementedError

    def clip_v(self, v):
        return np.maximum(np.minimum(v, self.v_max), self.v_min)

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

    def sample_uniform_v(self):
        """
        Samples an input uniformly at random from within the input bounds
        :return: uniformly random input
        """
        return np.random.uniform(self.v_min, self.v_max, size=(self.n_robots, self.v_max.shape[0]))

    @staticmethod
    def plot_spacial(ax, xt, c='-b'):
        """
        Plots the x, y spatial trajectory on the given axes
        :param ax: axes on which to plot
        :param xt: state trajectory
        :param c: color/line type
        """
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


class SingleInt2D(RomDynamics):
    n = 2   # [x, y]
    m = 2   # [vx, vy]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi'):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend)
        self.A = self.const_mat([[1.0, 0], [0, 1.0]])
        self.B = self.const_mat([[dt, 0], [0, dt]])

    def f(self, x, u):
        return (self.A @ x.T).T + (self.B @ u.T).T

    def proj_z(self, x):
        return x[..., :2]

    def des_pose_vel(self, z, v):
        return self.stack((z, self.arctan2(v[:, 1], v[:, 0])[:, None])), self.stack((v, self.zero_mat(v.shape[0], 1)))

    def sample_uniform_bounded_v(self, z):
        return self.sample_uniform_v()

    def clip_v_z(self, z, v):
        return v

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y'])
        axs[1].legend(['vx', 'vy'])


class DoubleInt2D(RomDynamics):
    n = 4   # [x, y, vx, vy]
    m = 2   # [ax, ay]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi'):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend)
        self.A = self.const_mat([[1.0, 0, dt, 0], [0, 1.0, 0, dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
        self.B = self.const_mat([[0, 0], [0, 0], [dt, 0], [0, dt]])

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
        v_max_z = np.minimum(self.v_max, (self.z_max[2:] - z[:, 2:]) / self.dt)
        v_min_z = np.maximum(self.v_min, (self.z_min[2:] - z[:, 2:]) / self.dt)
        return v_min_z, v_max_z

    def sample_uniform_bounded_v(self, z):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        return np.random.uniform(v_min_z, v_max_z)

    def clip_v_z(self, z, v):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        return np.maximum(np.minimum(v, v_max_z), v_min_z)

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'vx', 'vy'])
        axs[1].legend(['ax', 'ay'])


class Unicycle(RomDynamics):
    n = 3   # [x, y, theta]
    m = 2   # [v, omega]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi'):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend)

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
        return z[:, :3], self.f(z, v)[:, :3]

    def sample_uniform_bounded_v(self, z):
        return self.sample_uniform_v()

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


class LateralUnicycle(Unicycle):
    n = 3   # [x, y, theta]
    m = 3   # [v, v_perp, omega]

    def f(self, x, u):
        gu = self.zero_mat(self.n_robots, self.n)
        gu[:, 0] = u[:, 0] * self.cos(x[:, 2]) - u[:, 1] * self.sin(x[:, 2])
        gu[:, 1] = u[:, 0] * self.sin(x[:, 2]) + u[:, 1] * self.cos(x[:, 2])
        gu[:, 2] = u[:, 2]
        return x + self.dt * gu

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta'])
        axs[1].legend(['v', 'v_perp', 'omega'])


class ExtendedUnicycle(Unicycle):
    n = 5   # [x, y, theta, v, omega]
    m = 2   # [a, alpha]

    def f(self, x, u):
        gu = self.zero_mat(self.n_robots, self.n)
        gu[:, 0] = x[:, 3] * self.cos(x[:, 2])
        gu[:, 1] = x[:, 3] * self.sin(x[:, 2])
        gu[:, 2] = x[:, 4]
        gu[:, 3] = u[:, 0]
        gu[:, 4] = u[:, 1]
        return x + self.dt * gu

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
        v_max_z = np.minimum(self.v_max, (self.z_max[3:] - z[:, 3:]) / self.dt)
        v_min_z = np.maximum(self.v_min, (self.z_min[3:] - z[:, 3:]) / self.dt)
        return v_min_z, v_max_z

    def sample_uniform_bounded_v(self, z):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        return np.random.uniform(v_min_z, v_max_z)

    def clip_v_z(self, z, v):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        return np.maximum(np.minimum(v, v_max_z), v_min_z)

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta', 'v', 'omega'])
        axs[1].legend(['a', 'alpha'])


class ExtendedLateralUnicycle(ExtendedUnicycle):
    n = 6   # [x, y, theta, v, v_perp, omega]
    m = 3   # [a, a_perp, alpha]

    def f(self, x, u):
        gu = self.zero_mat(self.n_robots, self.n)
        gu[:, 0] = x[:, 3] * self.cos(x[:, 2]) - x[:, 4] * self.sin(x[:, 2])
        gu[:, 1] = x[:, 3] * self.sin(x[:, 2]) + x[:, 4] * self.cos(x[:, 2])
        gu[:, 2] = x[:, 5]
        gu[:, 3] = u[:, 0]
        gu[:, 4] = u[:, 1]
        gu[:, 5] = u[:, 2]
        return x + self.dt * gu

    def proj_z(self, x):
        quat = x[:, 3:7]
        v = x[:, 7:9]
        rot = Rotation.from_quat(quat)
        eul = rot.as_euler('xyz', degrees=False)
        v_local = np.squeeze(yaw2rot(eul[..., -1]) @ v[:, :, None])
        return self.stack((x[..., :2], eul[..., -1][:, None], v_local, x[:, -1][:, None]))

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta', 'v', 'v_perp', 'omega'])
        axs[1].legend(['a', 'a_perp', 'alpha'])
