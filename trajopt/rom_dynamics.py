import numpy as np
import casadi as ca
from abc import ABC, abstractmethod


class RomDynamics(ABC):
    n: int
    m: int

    def __init__(self, dt, z_min, z_max, v_min, v_max, backend='casadi'):
        self.dt = dt
        self.v_min = v_min
        self.v_max = v_max
        self.z_min = z_min
        self.z_max = z_max
        if backend == 'casadi':
            self.zero_mat = lambda r, c: ca.MX(r, c)
            self.zero_vec = lambda n: ca.MX(n, 1)
            self.const_mat = lambda m: ca.DM(m)
            self.sin = lambda x: ca.sin(x)
            self.cos = lambda x: ca.cos(x)
        elif backend == 'numpy':
            self.zero_mat = lambda r, c: np.zeros((r, c))
            self.zero_vec = lambda n: np.zeros((n,))
            self.const_mat = lambda m: np.array(m)
            self.sin = lambda x: np.sin(x)
            self.cos = lambda x: np.cos(x)

    @abstractmethod
    def f(self, x, u):
        raise NotImplementedError

    @abstractmethod
    def sample_uniform_bounded_v(self, z):
        raise NotImplementedError

    @abstractmethod
    def clip_v(self, z, v):
        raise NotImplementedError

    def sample_uniform_v(self):
        return np.random.uniform(self.v_min, self.v_max)

    @staticmethod
    def plot_spacial(ax, xt, c='-b'):
        ax.plot(xt[:, 0], xt[:, 1], c)

    def plot_ts(self, axs, xt, ut):
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

    def __init__(self, dt, z_min, z_max, v_min, v_max, backend='casadi'):
        super().__init__(dt, z_min, z_max, v_min, v_max, backend=backend)
        self.A = self.const_mat([[1.0, 0], [0, 1.0]])
        self.B = self.const_mat([[dt, 0], [0, dt]])

    def f(self, x, u):
        return self.A @ x + self.B @ u

    def sample_uniform_bounded_v(self, z):
        return self.sample_uniform_v()

    def clip_v(self, z, v):
        return v

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y'])
        axs[1].legend(['vx', 'vy'])


class DoubleInt2D(RomDynamics):
    n = 4   # [x, y, vx, vy]
    m = 2   # [ax, ay]

    def __init__(self, dt, z_min, z_max, v_min, v_max, backend='casadi'):
        super().__init__(dt, z_min, z_max, v_min, v_max, backend=backend)
        self.A = self.const_mat([[1.0, 0, dt, 0], [0, 1.0, 0, dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
        self.B = self.const_mat([[0, 0], [0, 0], [dt, 0], [0, dt]])

    def f(self, x, u):
        return self.A @ x + self.B @ u

    def compute_state_dependent_v_bounds(self, z):
        v_max_z = np.minimum(self.v_max, (self.z_max[2:] - z[2:]) / self.dt)
        v_min_z = np.maximum(self.v_min, (self.z_min[2:] - z[2:]) / self.dt)
        return v_min_z, v_max_z

    def sample_uniform_bounded_v(self, z):
        v_min_z, v_max_z = self.compute_state_dependent_v_bounds(z)
        return np.random.uniform(v_min_z, v_max_z)

    def clip_v(self, z, v):
        v_min_z, v_max_z = self.compute_state_dependent_v_bounds(z)
        return np.maximum(np.minimum(v, v_max_z), v_min_z)

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'vx', 'vy'])
        axs[1].legend(['ax', 'ay'])


class Unicycle(RomDynamics):
    n = 3   # [x, y, theta]
    m = 2   # [v, omega]

    def __init__(self, dt, z_min, z_max, v_min, v_max, backend='casadi'):
        super().__init__(dt, z_min, z_max, v_min, v_max, backend=backend)

    def f(self, x, u):
        g = self.zero_mat(self.n, self.m)
        g[0, 0] = self.cos(x[2])
        g[1, 0] = self.sin(x[2])
        g[2, 1] = 1.0
        return x + self.dt * g @ u

    def sample_uniform_bounded_v(self, z):
        return self.sample_uniform_v()

    def clip_v(self, z, v):
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
        g = self.zero_mat(self.n, self.m)
        g[0, 0] = self.cos(x[2])
        g[0, 1] = -self.sin(x[2])
        g[1, 0] = self.sin(x[2])
        g[1, 1] = self.cos(x[2])
        g[2, 2] = 1.0
        return x + self.dt * g @ u

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta'])
        axs[1].legend(['v', 'v_perp', 'omega'])


class ExtendedUnicycle(Unicycle):
    n = 5   # [x, y, theta, v, omega]
    m = 2   # [a, alpha]

    def f(self, x, u):
        gu = self.zero_vec(self.n)
        gu[0] = x[3] * self.cos(x[2])
        gu[1] = x[3] * self.sin(x[2])
        gu[2] = x[4]
        gu[3] = u[0]
        gu[4] = u[1]
        return x + self.dt * gu

    def compute_state_dependent_v_bounds(self, z):
        v_max_z = np.minimum(self.v_max, (self.z_max[3:] - z[3:]) / self.dt)
        v_min_z = np.maximum(self.v_min, (self.z_min[3:] - z[3:]) / self.dt)
        return v_min_z, v_max_z

    def sample_uniform_bounded_v(self, z):
        v_min_z, v_max_z = self.compute_state_dependent_v_bounds(z)
        return np.random.uniform(v_min_z, v_max_z)

    def clip_v(self, z, v):
        v_min_z, v_max_z = self.compute_state_dependent_v_bounds(z)
        return np.maximum(np.minimum(v, v_max_z), v_min_z)

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta', 'v', 'omega'])
        axs[1].legend(['a', 'alpha'])


class ExtendedLateralUnicycle(ExtendedUnicycle):
    n = 6   # [x, y, theta, v, v_perp, omega]
    m = 3   # [a, a_perp, alpha]

    def f(self, x, u):
        gu = self.zero_vec(self.n)
        gu[0] = x[3] * self.cos(x[2]) - x[4] * self.sin(x[2])
        gu[1] = x[3] * self.sin(x[2]) + x[4] * self.cos(x[2])
        gu[2] = x[5]
        gu[3] = u[0]
        gu[4] = u[1]
        gu[5] = u[2]
        return x + self.dt * gu

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta', 'v', 'v_perp', 'omega'])
        axs[1].legend(['a', 'a_perp', 'alpha'])
