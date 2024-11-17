import os
import wandb
import torch
from torch.utils.data import Dataset
from pathlib import Path

class CheckPointManager:
    def __init__(self, metric_name="loss"):
        self.metric_name = metric_name
        self.best_loss = float("inf")
        self.ckpt_path = str(Path(__file__).parent / "models" / f"{wandb.run.id}")
        os.makedirs(self.ckpt_path, exist_ok=True)

    def save(self, model, metric, epoch, step):
        self._model_save(model)

        if metric < self.best_loss:
            self.best_loss = metric
            self._model_save(model, True)

    def _model_save(self, model, best=False):
        prefix = "best" if best else "latest"
        torch.save(model.state_dict(), f"{self.ckpt_path}/{prefix}_model.pth")

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