import torch
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
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    y2r = np.zeros((yaw.shape[0], 2, 2))
    y2r[:, 0, 0] = cy
    y2r[:, 0, 1] = sy
    y2r[:, 1, 0] = -sy
    y2r[:, 1, 1] = cy
    return y2r


def wrap_angles(ang):
    """
    Wraps angles to [-pi, pi]
    :param ang: angles to wrap
    :return: wrapped angles
    """
    return ((ang + np.pi) % (2 * np.pi)) - np.pi


def unnormalize_dict(normalized_dict, sep="/"):
    result = {}
    for key, value in normalized_dict.items():
        keys = key.split(sep)
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result


def evaluate_scalar_tube(test_dataset, loss_fn, device):

    def eval_model(model):
        model.eval()
        metrics = {}

        with torch.no_grad():
            data, w = test_dataset.data.to(device), test_dataset.target.to(device)

            fw = model(data)
            test_loss = loss_fn(fw, w, data)

            correct_mask = fw > w
            differences = (w[correct_mask] - fw[correct_mask]).abs()

            metrics[f'Test Loss (alpha={loss_fn.alpha:.1f})'] = test_loss
            metrics[f'Proportion Correct, fw > w (alpha={loss_fn.alpha:.1f})'] = correct_mask.float().mean()
            metrics[f'Mean Error when Correct, fw > w (alpha={loss_fn.alpha:.1f})'] = differences.mean()

        return metrics

    return eval_model


def evaluate_error_dynamics(test_dataset, loss_fn, device):

    def eval_model(model):
        model.eval()
        metrics = {}

        with torch.no_grad():
            data, e = test_dataset.data.to(device), test_dataset.target.to(device)

            fe = model(data)
            test_loss = loss_fn(fe, e, data)

            metrics[f'Test Loss'] = test_loss

        return metrics

    return eval_model

