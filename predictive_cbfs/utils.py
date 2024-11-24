import os
import wandb
import torch
from torch.utils.data import Dataset
from pathlib import Path
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from legged_gym.utils.helpers import torch_rand_vec_float

from deep_tube_learning.utils import torch_rand_vec_float


class CheckPointManager:
    def __init__(self, run_id, metric_name="loss"):
        self.metric_name = metric_name
        self.best_loss = float("inf")
        self.ckpt_path = str(Path(__file__).parent / "models" / f"{run_id}")
        os.makedirs(self.ckpt_path, exist_ok=True)

    def save(self, model, metric, epoch, step):
        self._model_save(model, epoch)

        if metric < self.best_loss:
            self.best_loss = metric
            self._model_save(model, epoch, True)

    def _model_save(self, model, epoch, best=False):
        prefix = "best" if best else "latest"
        torch.save(model.state_dict(), f"{self.ckpt_path}/{prefix}_model_{epoch}.pth")

    def to_wandb(self):
        artifact = wandb.Artifact(
            type="model",
            name=f"{wandb.run.id}_model"
        )
        artifact.add_dir(str(self.ckpt_path))
        wandb.run.log_artifact(artifact)


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def eta_schedule(epoch, eta_max, eta_decay):
    return eta_max * eta_decay**epoch


def pretrain_step_conservative(env, policy, cfg, pretrain_size=4000000):
    obs, _ = env.reset()

    lb = torch.tensor([
        env.default_root_pos_noise_lower[0],
        env.default_root_pos_noise_lower[1],
        0.2,
        -0.35,
        -0.35,
        -4
    ], device=obs.device)
    ub = torch.tensor([
        env.default_root_pos_noise_upper[0],
        env.default_root_pos_noise_upper[1],
        0.6,
        0.35,
        0.35,
        3
    ], device=obs.device)
    # Randomize other states
    x = torch_rand_vec_float(lb, ub, (pretrain_size, 6), obs.device)
    delta_targ = torch.ones((pretrain_size, 1)).to(obs.device) * policy.v_filt.delta_max
    optimizer = instantiate(cfg.optimizer)(policy.v_filt.delta.parameters())
    lr_scheduler = instantiate(cfg.lr_scheduler)(optimizer)
    loss_fn = instantiate(cfg.loss_fn)
    loader = DataLoader(RegressionDataset(x, delta_targ), batch_size=cfg.batch_size, shuffle=True)

    for t_ep in range(cfg.train_epochs):
        policy.v_filt.delta.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {t_ep}/{cfg.train_epochs}")
        for batch in pbar:
            data, targets = batch[0].to(env.device), batch[1].to(env.device)
            optimizer.zero_grad()
            outputs = policy.v_filt.delta(data)
            loss = loss_fn(outputs, targets)

            loss.backward()
            if 'grad_clip' in cfg.keys():
                clip_grad_norm_(policy.v_filt.delta.parameters(), cfg.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()

            # Compute gradient norm
            grads = [
                param.grad.detach().flatten()
                for param in policy.v_filt.delta.parameters()
                if param.grad is not None
            ]
            grad_norm = torch.cat(grads).norm()

            pbar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
            epoch_loss += loss