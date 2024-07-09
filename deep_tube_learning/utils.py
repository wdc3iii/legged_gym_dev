import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation


class AbstractSampleHoldDT(ABC):

    @abstractmethod
    def sample(self):
        raise NotImplementedError


class UniformSampleHoldDT:

    def __init__(self, t_low, t_high, n_robots=1):
        self.t_low = t_low
        self.t_high = t_high
        self.n_robots = n_robots

    def sample(self):
        return np.random.randint(self.t_low, self.t_high, size=(self.n_robots,))


def quat2yaw(quat):
    rot = Rotation.from_quat(quat)
    eul = rot.as_euler('xyz', degrees=False)
    return eul[:, -1]


def yaw2rot(yaw):
    return np.stack([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]).reshape((-1, 2, 2))

