import numpy as np
import casadi as ca
from abc import ABC, abstractmethod


class RomDynamics(ABC):
    n: int
    m: int

    def __init__(self, dt):
        self.dt = dt

    @abstractmethod
    def f(self, x, u):
        raise NotImplementedError

    @staticmethod
    def plot_spacial(ax, xt, c='-k'):
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


class DoubleInt2D(RomDynamics):
    n = 4   # [x, y, vx, vy]
    m = 2   # [ax, ay]

    def __init__(self, dt):
        super().__init__(dt)
        self.A = ca.DM([[1.0, 0, dt, 0], [0, 1.0, 0, dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
        self.B = ca.DM([[0, 0], [0, 0], [dt, 0], [0, dt]])

    def f(self, x, u):
        return self.A @ x + self.B @ u

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'vx', 'vy'])
        axs[1].legend(['ax', 'ay'])


class SingleInt2D(RomDynamics):
    n = 2   # [x, y]
    m = 2   # [vx, vy]

    def __init__(self, dt):
        super().__init__(dt)
        self.A = ca.DM([[1.0, 0], [0, 1.0]])
        self.B = ca.DM([[dt, 0], [0, dt]])

    def f(self, x, u):
        return self.A @ x + self.B @ u

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y'])
        axs[1].legend(['vx', 'vy'])


class Unicycle(RomDynamics):
    n = 3   # [x, y, theta]
    m = 2   # [v, omega]

    def __init__(self, dt):
        super().__init__(dt)

    def f(self, x, u):
        g = ca.MX(self.n, self.m)
        g[0, 0] = ca.cos(x[2])
        g[1, 0] = ca.sin(x[2])
        g[2, 1] = 1.0
        return x + self.dt * g @ u

    @staticmethod
    def plot_spacial(ax, xt, c='-k'):
        RomDynamics.plot_spacial(ax, xt, c=c)
        ax.quiver(xt[:, 0], xt[:, 1], np.cos(xt[:, 2]), np.sin(xt[:, 2]))

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta'])
        axs[1].legend(['v', 'omega'])


class LateralUnicycle(Unicycle):
    n = 3   # [x, y, theta]
    m = 3   # [v, v_perp, omega]

    def __init__(self, dt):
        super().__init__(dt)

    def f(self, x, u):
        g = ca.MX(self.n, self.m)
        g[0, 0] = ca.cos(x[2])
        g[0, 1] = -ca.sin(x[2])
        g[1, 0] = ca.sin(x[2])
        g[1, 1] = ca.cos(x[2])
        g[2, 2] = 1.0
        return x + self.dt * g @ u

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta'])
        axs[1].legend(['v', 'v_perp', 'omega'])


class ExtendedUnicycle(Unicycle):
    n = 5   # [x, y, theta, v, omega]
    m = 2   # [a, alpha]

    def __init__(self, dt):
        super().__init__(dt)

    def f(self, x, u):
        gu = ca.MX(self.n, 1)
        gu[0, 0] = x[3] * ca.cos(x[2])
        gu[1, 0] = x[3] * ca.sin(x[2])
        gu[2, 0] = x[4]
        gu[3, 0] = u[0]
        gu[4, 0] = u[1]
        return x + self.dt * gu

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta', 'v', 'omega'])
        axs[1].legend(['a', 'alpha'])


class ExtendedLateralUnicycle(Unicycle):
    n = 6   # [x, y, theta, v, v_perp, omega]
    m = 3   # [a, a_perp, alpha]

    def __init__(self, dt):
        super().__init__(dt)

    def f(self, x, u):
        gu = ca.MX(self.n, 1)
        gu[0, 0] = x[3] * ca.cos(x[2]) - x[4] * ca.sin(x[2])
        gu[1, 0] = x[3] * ca.sin(x[2]) + x[4] * ca.cos(x[2])
        gu[2, 0] = x[5]
        gu[3, 0] = u[0]
        gu[4, 0] = u[1]
        gu[5, 0] = u[2]
        return x + self.dt * gu

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y', 'theta', 'v', 'v_perp', 'omega'])
        axs[1].legend(['a', 'a_perp', 'alpha'])
