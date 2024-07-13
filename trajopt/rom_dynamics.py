import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
from deep_tube_learning.utils import yaw2rot
import matplotlib.cm as cm


class RomDynamics(ABC):
    """
    Abstract class for Reduced order Model Dynamics
    """
    n: int  # Dimension of state
    m: int  # Dimension of input

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi', methods=['uniform'], weights=[1.0], seed=42):
        """
        Common constructor functionality
        :param dt: time discretization
        :param z_min: lower state bound
        :param z_max: upper state bound
        :param v_min: lower input bound
        :param v_max: upper input bound
        :param backend: 'casadi' for when using dynamics for a casadi optimization program,
               'numpy' for use with numpy arrays
        :param methods: list of methods to generate trajectories
        :param weights: weights for linear combination of trajectories
        """
        self.dt = dt
        self.v_min = v_min
        self.v_max = v_max
        self.z_min = z_min
        self.z_max = z_max
        self.n_robots = n_robots
        self.methods = methods
        self.weights = weights
        self.precomputed_v = None
        self.rng = np.random.RandomState(seed)
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
    def generate_and_store_trajectory(self, z, method, length):
        """
        Generates and stores a trajectory
        :param z: initial state
        :param method: method to generate trajectory
        :param length: length of the trajectory
        """
        raise NotImplementedError

    @abstractmethod
    def sample_uniform_bounded_v(self, z, length):
        """
        Samples an input, v which respects input bounds, and, when the dynamics are applied,
        will not result in velocities which violate the state bounds (when applicable)
        :param length: length of trajectory
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
        return self.rng.uniform(self.v_min, self.v_max, size=(self.n_robots, self.v_max.shape[0]))

    @staticmethod
    def plot_spacial(ax, xt, c='-b'):
        """
        Plots the x, y spatial trajectory on the given axes with a color gradient to indicate time series.
        :param ax: axes on which to plot
        :param xt: state trajectory
        :param c: color/line type
        """
        N = xt.shape[0]
        colors = cm.viridis(np.linspace(0, 1, N))  # Use the 'viridis' colormap

        # Plot segments with color gradient
        for i in range(N - 1):
            ax.plot(xt[i:i + 2, 0], xt[i:i + 2, 1], color=colors[i])
        scatter = ax.scatter(xt[:, 0], xt[:, 1], c=np.linspace(0, 1, N), cmap='viridis', s=10,
                             edgecolor='none')  # Plot points for better visibility

        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time')

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

    def sample_ramp_v(self, from_v, to_v, t_start, t_end, t):
        """
        Generates a ramping velocity from `from_v` to `to_v` between `t_start` and `t_end`.
        """
        ramp_v = np.zeros_like(from_v)
        for i in range(len(from_v)):
            if t < t_start:
                ramp_v[i] = from_v[i]
            elif t > t_end:
                ramp_v[i] = to_v[i]
            else:
                ramp_v[i] = from_v[i] + (to_v[i] - from_v[i]) * ((t - t_start) / (t_end - t_start))
        return ramp_v

    def sample_sin_v(self, amplitude, frequency, c, t_final):
        """
        Generates a sinusoidal velocity with given amplitude, frequency, and offset.
        :param amplitude: Amplitude of the sine wave
        :param frequency: Frequency of the sine wave
        :param c: Offset of the sine wave
        :param t_final: Final time
        :return: Function to generate sinusoidal velocities
        """
        def sin_func(t):
            return np.array([amplitude[i] * self.sin(2 * np.pi * frequency[i] * t) + c[i] for i in range(len(amplitude))])
        return sin_func

    def sample_extreme_v(self, z):
        possible_values = [self.v_min, self.v_max, np.zeros_like(self.v_min)]
        return np.array([[self.rng.choice([vmin, vmax, 0]) for vmin, vmax in zip(self.v_min, self.v_max)] for _ in
                         range(self.n_robots)])

    def generate_and_store_multiple_trajectories(self, z, methods, length, weights):
        """
        Generates multiple trajectories and combines them using the given weights
        :param z: initial state
        :param methods: list of methods to generate trajectories
        :param length: length of the trajectory
        :param weights: weights for linear combination of trajectories
        """
        trajectories = np.zeros((len(methods), length, self.n_robots, self.m))
        for i, method in enumerate(methods):
            self.generate_and_store_trajectory(z, method, length)
            trajectories[i] = self.precomputed_v

        self.precomputed_v = np.tensordot(weights, trajectories, axes=([0], [0]))


class SingleInt2D(RomDynamics):
    n = 2   # [x, y]
    m = 2   # [vx, vy]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi',
                 methods=['uniform'], weights=[1.0], seed=42):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend,
                         methods=methods, weights=weights, seed=seed)
        self.A = self.const_mat([[1.0, 0], [0, 1.0]])
        self.B = self.const_mat([[dt, 0], [0, dt]])

    def f(self, x, u):
        return (self.A @ x.T).T + (self.B @ u.T).T

    def generate_and_store_trajectory(self, z, method, length):
        if method == 'uniform':
            self.precomputed_v = self.sample_uniform_bounded_v(z, length)
        elif method == 'ramp':
            self.precomputed_v = self.sample_ramp_bounded_v(z, length)
        elif method == 'sin':
            self.precomputed_v = self.sample_sin_bounded_v(z, length)
        elif method == 'extreme':
            self.precomputed_v = self.sample_extreme_bounded_v(z, length)

    def proj_z(self, x):
        return x[..., :2]

    def des_pose_vel(self, z, v):
        return self.stack((z, self.arctan2(v[:, 1], v[:, 0])[:, None])), self.stack((v, self.zero_mat(v.shape[0], 1)))

    def sample_uniform_bounded_v(self, z, length):
        return np.tile(self.sample_uniform_v(), (length, 1, 1))

    def sample_extreme_bounded_v(self, z, length):
        extreme_v = np.array([self.sample_extreme_v(z) for _ in range(length)])
        return extreme_v

    def sample_sin_bounded_v(self, z, length):
        amplitude = np.random.uniform(low=self.v_min, high=self.v_max, size=(self.n_robots, 2))
        frequency = np.random.uniform(low=0.5, high=1.5, size=(self.n_robots, 2))
        c = np.random.uniform(low=(amplitude - self.v_min), high=(amplitude - self.v_max), size=(self.n_robots, 2))
        t_final = length * self.dt
        sin_func = self.sample_sin_v(amplitude, frequency, c, t_final)
        return np.array([sin_func(t * self.dt) for t in range(length)])

    def sample_ramp_bounded_v(self, z, length):
        from_v = np.random.uniform(low=self.v_min, high=self.v_max, size=(self.n_robots, 2))
        to_v = np.random.uniform(low=self.v_min, high=self.v_max, size=(self.n_robots, 2))
        t_start = 0.0
        t_end = length * self.dt
        ramp_v = np.array([self.sample_ramp_v(from_v, to_v, t_start, t_end, t * self.dt) for t in range(length)])
        return ramp_v

    def clip_v_z(self, z, v):
        return v

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y'])
        axs[1].legend(['vx', 'vy'])


class DoubleInt2D(RomDynamics):
    n = 4   # [x, y, vx, vy]
    m = 2   # [ax, ay]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi', method='uniform', seed=42):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend, method=method, seed=seed)
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


    def sample_uniform_bounded_v(self, z, length):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        return np.tile(self.rng.uniform(v_min_z, v_max_z), (length, 1, 1))

    def clip_v_z(self, z, v):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        return np.maximum(np.minimum(v, v_max_z), v_min_z)

    def generate_and_store_trajectory(self, z, method, length):
        if self.method == 'uniform':
            self.precomputed_v = self.sample_uniform_bounded_v(z, length)
        elif self.method == 'ramp':
            self.precomputed_v = self.sample_ramp_bounded_v(z, length)
        elif self.method == 'sin':
            self.precomputed_v = self.sample_sin_bounded_v(z, length)
        elif self.method == 'extreme':
            self.precomputed_v = self.sample_extreme_bounded_v(z, length)

    def sample_extreme_bounded_v(self, z, length):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        possible_values = [v_min_z, v_max_z, np.zeros_like(v_min_z)]
        extreme_samples = np.array([[self.rng.choice([v_min_z[j, i], v_max_z[j, i], 0])
                                     for i in range(v_min_z.shape[1])]
                                    for j in range(v_min_z.shape[0])])
        return np.tile(extreme_samples, (length, 1, 1))

    def sample_sin_bounded_v(self, z, length):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        amplitude = np.random.uniform(low=v_min_z, high=v_max_z, size=(self.n_robots, 2))
        frequency = np.random.uniform(low=0.5, high=1.5, size=(self.n_robots, 2))
        c = np.random.uniform(low=(amplitude - v_min_z), high=(amplitude - v_max_z), size=(self.n_robots, 2))  # Randomize offset values within acceptable range
        t_final = length * self.dt
        sin_func = self.sample_sin_v(amplitude, frequency, c, t_final)
        return np.array([sin_func(t * self.dt) for t in range(length)])

    def sample_ramp_bounded_v(self, z, length):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        from_v = np.random.uniform(low=v_min_z, high=v_max_z, size=(self.n_robots, 2))
        to_v = np.random.uniform(low=v_min_z, high=v_max_z, size=(self.n_robots, 2))
        t_start = 0.0
        t_end = length * self.dt
        return np.array([[self.sample_ramp_v(from_v[i], to_v[i], t_start, t_end, t * self.dt) for i in
                         range(self.n_robots)] for t in range(length)])

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'vx', 'vy'])
        axs[1].legend(['ax', 'ay'])


class Unicycle(RomDynamics):
    n = 3   # [x, y, theta]
    m = 2   # [v, omega]

    def __init__(self, dt, z_min, z_max, v_min, v_max, n_robots=1, backend='casadi', method='uniform', seed=42):
        super().__init__(dt, z_min, z_max, v_min, v_max, n_robots=n_robots, backend=backend, method=method, seed=seed)

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

    def sample_uniform_bounded_v(self, z, length):
        return np.tile(self.sample_uniform_v(), (length, 1, 1))

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

    def generate_and_store_trajectory(self, z, method, length):
        if method == 'uniform':
            self.precomputed_v = self.sample_uniform_bounded_v(z, length)
        elif method == 'ramp':
            self.precomputed_v = self.sample_ramp_bounded_v(z, length)
        elif method == 'sin':
            self.precomputed_v = self.sample_sin_bounded_v(z, length)
        elif method == 'extreme':
            self.precomputed_v = self.sample_extreme_bounded_v(z, length)

    def sample_extreme_bounded_v(self, z, length):
        return np.tile(self.sample_extreme_v(z), (length, 1, 1))

    def sample_sin_bounded_v(self, z, length):
        amplitude = np.random.uniform(low=self.v_min, high=self.v_max, size=2)
        frequency = np.random.uniform(low=0.5, high=1.5, size=2)
        c = np.random.uniform(low=(amplitude - self.v_min), high=(amplitude - self.v_max), size=2)  # Randomize offset values within acceptable range
        t_final = length * self.dt
        sin_func = self.sample_sin_v(amplitude, frequency, c, t_final)
        return np.array([[sin_func(t * self.dt)] for t in range(length)])

    def sample_ramp_bounded_v(self, z, length):
        from_v = np.random.uniform(low=self.v_min, high=self.v_max, size=(self.n_robots, self.m))
        to_v = np.random.uniform(low=self.v_min, high=self.v_max, size=(self.n_robots, self.m))
        t_start = 0.0
        t_end = length * self.dt
        return np.array([[self.sample_ramp_v(from_v[i], to_v[i], t_start, t_end, t * self.dt) for i in
                         range(self.n_robots)] for t in range(length)])

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
        v_max_z = np.minimum(self.v_max, (self.z_max[3:] - z[:, 3:]) / self.dt)
        v_min_z = np.maximum(self.v_min, (self.z_min[3:] - z[:, 3:]) / self.dt)
        return v_min_z, v_max_z

    def sample_uniform_bounded_v(self, z):
        v_min_z, v_max_z = self.compute_state_dependent_input_bounds(z)
        return self.rng.uniform(v_min_z, v_max_z)

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
        v_local = np.squeeze(yaw2rot(eul[..., -1]) @ v[:, :, None])
        return self.stack((x[..., :2], eul[..., -1][:, None], v_local, x[:, -1][:, None]))

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta', 'v', 'v_perp', 'omega'])
        axs[1].legend(['a', 'a_perp', 'alpha'])
