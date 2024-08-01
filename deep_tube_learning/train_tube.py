import os

import hydra
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from deep_tube_learning.datasets import TubeDataset


class CheckPointManager:
    def __init__(self, metric_name="loss"):
        self.metric_name = metric_name
        self.best_loss = float("inf")
        self.ckpt_path = str(Path(__file__).parent / "models" / f"{wandb.run.id}")
        os.makedirs(self.ckpt_path, exist_ok=True)

    def save(self, model, metric, epoch, step):
        self._model_save(model)
        artifact = wandb.Artifact(
            type="model",
            name=f"{wandb.run.id}_model",
            metadata={self.metric_name: metric, "epoch": epoch, "step": step},
        )

        artifact.add_dir(str(self.ckpt_path))

        aliases = ["latest"]

        if self.best_loss > metric:
            self.best_loss = metric
            aliases.append("best")

        wandb.run.log_artifact(artifact, aliases=aliases)

    def _model_save(self, model):
        torch.save(model.state_dict(), f"{self.ckpt_path}/model.pth")


def create_data_loaders(dataset: TubeDataset, batch_size, validation_split):
    train_dataset, test_dataset = dataset.random_split(validation_split)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_dataset


@hydra.main(
    config_path=str(Path(__file__).parent / "configs" / "tube_learning"),
    config_name="tube_learning",
    version_base="1.2",
)
def main(cfg):
    # Seed RNG
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Set torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = instantiate(cfg.dataset)
    input_dim = dataset.input_dim
    output_dim = dataset.output_dim

    loader, test_dataset = create_data_loaders(dataset, cfg.batch_size, cfg.validation_split)

    loss_fn = instantiate(cfg.loss)
    model = instantiate(cfg.model)(input_dim, output_dim)
    model.to(device)
    model_evaluation = instantiate(cfg.model_evaluation)(test_dataset, loss_fn, device)

    optimizer = instantiate(cfg.optimizer)(model.parameters())
    lr_scheduler = instantiate(cfg.lr_scheduler)(optimizer)

    # Send config to wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]
    wandb.init(project="Deep_Tube_Training",
               entity="coleonguard-Georgia Institute of Technology",
               name=cfg.experiment_name,
               config=cfg_dict)

    ckpt_manager = CheckPointManager(metric_name="loss")

    step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.num_epochs}")
        for batch in pbar:
            step += 1
            data, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets, data)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()

            # Compute gradient norm
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            grad_norm = torch.cat(grads).norm()

            # Log loss, lr, and gradient norm
            wandb.log(
                {
                    "loss_step": loss.item(),
                    "lr_step": lr_scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm
                },
                step=step,
            )
            pbar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
            epoch_loss += loss
            if step % cfg.steps_per_model_checkpoint == 0:
                ckpt_manager.save(model, loss.item(), epoch=epoch, step=step)
            if step % cfg.steps_per_model_evaluation == 0:
                metrics = model_evaluation(model)
                wandb.log(metrics, step=step)

        wandb.log(
            {"loss_epoch": epoch_loss.item() / len(loader), "lr_epoch": lr_scheduler.get_last_lr()[0]},
            step=step,
        )
        dataset.update()


if __name__ == "__main__":
    main()
