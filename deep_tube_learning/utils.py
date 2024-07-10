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


def evaluate_scalar_tube(test_dataset, loss_fn, device):

    def eval_model(model):
        model.eval()
        metrics = {}

        with torch.no_grad():
            data, targets = test_dataset.dataset.data.to(device), test_dataset.dataset.target.to(device)

            outputs = model(data)
            test_loss = loss_fn(outputs, targets, data)

            greater_mask = outputs > targets
            less_mask = outputs < targets
            differences = (targets[less_mask] - outputs[less_mask]).abs()


            metrics[f'Test Loss (alpha={loss_fn.alpha:.1f})'] = test_loss
            metrics[f'Proportion y_pred > w_{{t+1}} (alpha={loss_fn.alpha:.1f})'] = greater_mask.float().mean()
            metrics[f'Avg Abs Diff y_pred < w_{{t+1}} (alpha={loss_fn.alpha:.1f})'] = differences.mean()

        return metrics

    return eval_model


