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


def unnormalize_dict(normalized_dict, sep="/"):
    """
    Unnormalize a dictionary with keys separated by a separator
    :param normalized_dict:
    :param sep:
    :return:
    """
    result = {}
    for key, value in normalized_dict.items():
        keys = key.split(sep)
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result