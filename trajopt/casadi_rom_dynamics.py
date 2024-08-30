import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
import matplotlib.cm as cm


class CasadiRomDynamics(ABC):
    """
    Abstract class for Reduced order Model Dynamics
    """
    n: int  # Dimension of state
    m: int  # Dimension of input
    state_names: list
    input_names: list

    def __init__(self, dt, z_min, z_max, v_min, v_max):
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
        self.state_names = []
        self.input_names = []

    @abstractmethod
    def f(self, z, v):
        """
        Dynamics function
        :param z: current state
        :param v: input
        :return: next state
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
        for i in range(1, xt.shape[0]):
            if not np.isnan(wt[i - 1]):
                xc = xt[i, 0]
                yc = xt[i, 1]
                circ = plt.Circle((xc, yc), wt[i - 1], color=c, fill=False)
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


class CasadiSingleInt2D(CasadiRomDynamics):
    n = 2   # [x, y]
    m = 2   # [vx, vy]

    def __init__(self, dt, z_min, z_max, v_min, v_max):
        super().__init__(dt, z_min, z_max, v_min, v_max)
        self.A = ca.DM([[1.0, 0], [0, 1.0]])
        self.B = ca.DM([[dt, 0], [0, dt]])
        self.vel_inds = ca.DM([False, False])
        self.state_names = ['x', 'y']
        self.input_name = ['v_x', 'v_y']

    def f(self, x, u):
        return self.A @ x + self.B @ u

    def plot_ts(self, axs, xt, ut):
        super().plot_ts(axs, xt, ut)
        axs[0].legend(['x', 'y'])
        axs[1].legend(['vx', 'vy'])
