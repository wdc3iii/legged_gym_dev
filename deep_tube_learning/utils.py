import os
import wandb
import torch
import statistics
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from omegaconf import OmegaConf, DictConfig, ListConfig
from scipy.spatial.transform import Rotation


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def torch_rand_vec_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


class AbstractSampleHoldDT(ABC):

    @abstractmethod
    def sample(self, num_samples: int):
        raise NotImplementedError


class UniformSampleHoldDT:

    def __init__(self, t_low, t_high, seed=42, backend='numpy', device='cuda'):
        self.t_low = t_low
        self.t_high = t_high
        if backend == 'numpy':
            self.rng = np.random.RandomState(seed)
            def uniform(l, h, n):
                self.rng.uniform(self.t_low, self.t_high, size=(n,))
        elif backend == 'torch':
            def uniform(l, h, n):
                return torch_rand_float(l, h, (n, 1), device).squeeze()

        self.uniform = uniform

    def sample(self, num_samples: int):
        return self.uniform(self.t_low, self.t_high, num_samples)


class UniformWeightSampler:

    def __init__(self, m, dim=5, device='cuda'):
        self.device = device
        self.dim = dim
        self.m = m

    def sample(self, num_samples: int):
        new_weights = torch.rand(size=(num_samples, self.m, self.dim), device=self.device)
        return new_weights / torch.sum(new_weights, dim=-1, keepdim=True)


class UniformWeightSamplerNoExtreme(UniformWeightSampler):

    def sample(self, num_samples: int):
        new_weights = super().sample(num_samples)
        new_weights[:, :, 2] = 0
        return new_weights / torch.sum(new_weights, dim=-1, keepdim=True)


class UniformWeightSamplerNoRamp(UniformWeightSampler):

    def sample(self, num_samples: int):
        new_weights = super().sample(num_samples)
        new_weights[:, :, 2] = 0
        return new_weights / torch.sum(new_weights, dim=-1, keepdim=True)


class UniformWeightSamplerBiasRampExtreme(UniformWeightSampler):

    def sample(self, num_samples: int):
        new_weights = super().sample(num_samples)
        m1, m2 = torch.nonzero(torch.rand((num_samples, self.m), device=self.device) < 0.05, as_tuple=True)
        new_weights[m1, m2, 0] = 0
        new_weights[m1, m2, 3] = 0
        new_weights[m1, m2, 4] = 0
        return new_weights / torch.sum(new_weights, dim=-1, keepdim=True)


class UniformWeightSamplerBiasRampExtremeExp(UniformWeightSampler):

    def sample(self, num_samples: int):
        new_weights = super().sample(num_samples)
        m1, m2 = torch.nonzero(torch.rand((num_samples, self.m), device=self.device) < 0.05, as_tuple=True)
        new_weights[m1, m2, 0] = 0
        new_weights[m1, m2, 3] = 0
        new_weights[m1, m2, 4] = 0
        m1, m2 = torch.nonzero(torch.rand((num_samples, self.m), device=self.device) < 0.05, as_tuple=True)
        new_weights[m1, m2, 0] = 0
        new_weights[m1, m2, 1] = 0
        new_weights[m1, m2, 2] = 0
        new_weights[m1, m2, 3] = 0
        new_weights += 1e-6
        return new_weights / torch.sum(new_weights, dim=-1, keepdim=True)


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

def wandb_model_load_cpu(api, artifact_name):
    config, artifact = wandb_load_artifact(api, artifact_name)

    dir_name = artifact.download(root=Path("/tmp/wandb_downloads"))
    state_dict = torch.load(str(Path(dir_name) / "model.pth"), map_location=torch.device('cpu'))
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

def evaluate_scalar_tube_oneshot(test_dataset, loss_fn, device):

    def eval_model(model):
        model.eval()
        metrics = {}

        with torch.no_grad():
            tmp_data, tmp_e = test_dataset[0]
            datas = torch.zeros((len(test_dataset), *tmp_data.shape), device=device)
            e = torch.zeros(((len(test_dataset), *tmp_e.shape)), device=device)
            for ii in range(len(test_dataset)):
                d, t = test_dataset[ii]
                datas[ii] = d.to(device)
                e[ii] = t.to(device)

            fw = model(datas)
            test_loss = loss_fn(fw, e, datas)

            correct_mask = fw > e
            differences = (e[correct_mask] - fw[correct_mask]).abs()

            metrics[f'Test Loss (alpha={loss_fn.alpha:.1f})'] = test_loss
            metrics[f'Proportion Correct, fw > e (alpha={loss_fn.alpha:.1f})'] = correct_mask.float().mean()
            metrics[f'Mean Error when Correct, fw > e (alpha={loss_fn.alpha:.1f})'] = differences.mean()

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


def convert_to_native(obj):
    if isinstance(obj, ListConfig):
        return [convert_to_native(item) for item in obj]
    else:
        return obj


def set_attributes_from_dict(obj, nested_dict):
    for key, value in nested_dict.items():
        if value is None:
            continue
        if isinstance(obj, dict):
            obj[key] = convert_to_native(value)
        elif isinstance(value, DictConfig):
            # Recursively set attributes for nested dictionaries
            nested_obj = getattr(obj, key)
            set_attributes_from_dict(nested_obj, value)
        else:
            # Set the attribute value for the current object
            setattr(obj, key, convert_to_native(value))


def update_cfgs_from_hydra(cfg, env_cfg, train_cfg):
    set_attributes_from_dict(env_cfg, cfg.env_config)
    set_attributes_from_dict(train_cfg, cfg.train_config)
    return env_cfg, train_cfg


def update_args_from_hydra(cfg, args):
    for key, val in cfg.args.items():
        setattr(args, key, val)
    return args


def set_matching_attributes_from_dict(cfg, new_cfg):
    for key, val in new_cfg.items():
        if key in cfg.keys():
            if isinstance(val, DictConfig):
                set_attributes_from_dict(cfg[key], val)
            else:
                cfg[key] = val


def update_hydra_cfg(new_cfg, cfg):
    set_matching_attributes_from_dict(cfg, new_cfg)
    return cfg


def policy_runner_wandb_callback(
        data,
        lr,
        mean_std,
        actor_critic_state_dict,
        optimizer_state_dict,
        device,
        curriculum,
        steps_per_model_checkpoint,
        checkpointer
):

    log_dict = {
        "Loss/value_function": data['mean_value_loss'],
        'Loss/surrogate': data['mean_surrogate_loss'],
        'Loss/learning_rate': lr,
        'Policy/mean_noise_std': mean_std.item(),
        'Train/curriculum': curriculum
    }
    if len(data['rewbuffer']) > 0:
        mean_reward = statistics.mean(data['rewbuffer'])
        log_dict['Train/mean_reward'] = mean_reward
        log_dict['Train/mean_episode_length'] = statistics.mean(data['lenbuffer'])
    else:
        mean_reward = -float("inf")

    if data['ep_infos']:
        for key in data['ep_infos'][0]:
            infotensor = torch.tensor([], device=device)
            for ep_info in data['ep_infos']:
                # handle scalar and zero dimensional tensor infos
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.Tensor([ep_info[key]])
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(device)))
            value = torch.mean(infotensor)
            log_dict['Episode/' + key] = value
    wandb.log(log_dict, step=data['it'])

    if data['it'] % steps_per_model_checkpoint == 0:
        checkpointer.save({'model_state_dict': actor_critic_state_dict,
                           'optimizer_state_dict': optimizer_state_dict,
                           'iter': data['it'],
                           'infos': None},
                          metric=mean_reward,
                          step=data['it'],
                          curriculum=curriculum)


class CheckPointManager:
    def __init__(self, metric_name="reward"):
        self.metric_name = metric_name
        self.best_reward = -float("inf")
        self.ckpt_path = str(Path(__file__).parent / "models" / f"{wandb.run.id}")
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.previous_curriculum = 0

    def save(self, model, metric, step, curriculum):
        self._model_save(model)
        artifact = wandb.Artifact(
            type="model",
            name=f"{wandb.run.id}_model",
            metadata={self.metric_name: metric, "step": step},
        )

        artifact.add_dir(str(self.ckpt_path))

        aliases = ["latest"]

        if curriculum != self.previous_curriculum:
            self.best_reward = -float("inf")
            self.previous_curriculum = curriculum
        if self.best_reward < metric:
            self.best_reward = metric
            aliases.append(f"best{curriculum}")

        wandb.run.log_artifact(artifact, aliases=aliases)

    def _model_save(self, model_dict):
        torch.save(model_dict, f"{self.ckpt_path}/model.pth")


def get_warmup_cosine_decay(warmup_steps, max_steps):

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warm-up
            return current_step / warmup_steps
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
            return 0.5 * (1 + np.cos(torch.pi * progress))

    return lr_lambda