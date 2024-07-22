import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


class AbstractSampleHoldDT(ABC):

    @abstractmethod
    def sample(self, num_samples: int):
        raise NotImplementedError


class UniformSampleHoldDT:

    def __init__(self, t_low, t_high, seed=42):
        self.t_low = t_low
        self.t_high = t_high
        self.rng = np.random.RandomState(seed)

    def sample(self, num_samples: int):
        return self.rng.uniform(self.t_low, self.t_high, size=(num_samples,))


class UniformWeightSampler:

    def __init__(self, dim=4, seed=42):
        self.rng = np.random.RandomState(seed)
        self.dim = dim

    def sample(self, num_samples: int):
        new_weights = self.rng.uniform(size=(num_samples, self.dim))
        return new_weights / np.sum(new_weights, axis=-1, keepdims=True)


class UniformWeightSamplerNoExtreme:

    def __init__(self, dim=4, seed=42):
        self.rng = np.random.RandomState(seed)
        self.dim = dim

    def sample(self, num_samples: int):
        new_weights = self.rng.uniform(size=(num_samples, self.dim))
        new_weights[:, 2] = 0
        return new_weights / np.sum(new_weights, axis=-1, keepdims=True)


class WeightSamplerSampleAndHold:

    def __init__(self, dim=4, seed=42):
        self.rng = np.random.RandomState(seed)
        self.dim = dim

    def sample(self, num_samples: int):
        new_weights = self.rng.uniform(size=(num_samples, self.dim))
        new_weights[:, 1:] = 0
        return new_weights / np.sum(new_weights, axis=-1, keepdims=True)


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



def wandb_model_load(api, artifact_name):
    config, artifact = wandb_load_artifact(api, artifact_name)

    dir_name = artifact.download(root=Path("/tmp/wandb_downloads"))
    state_dict = torch.load(str(Path(dir_name) / "model.pth"))
    return config, state_dict


def wandb_load_artifact(api, artifact_name):
    artifact = api.artifact(artifact_name)
    run = artifact.logged_by()
    config = run.config
    config = unnormalize_dict(config)
    config = OmegaConf.create(config)

    return config, artifact


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

