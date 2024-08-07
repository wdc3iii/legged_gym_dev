import hydra
from omegaconf import OmegaConf
import wandb
import itertools
import subprocess
import os
import time
from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from legged_gym import LEGGED_GYM_ROOT_DIR
from train_rl import main as train_rl_main

# Define the grid of hyperparameters to tune
param_grid = {
    'env_config.curriculum.rewards.tracking_rom': [[1.0, 0.8, 0.6], [1.2, 1.0, 0.8]],
    'env_config.curriculum.push.magnitude': [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]],
    'env_config.curriculum.push.time': [[13, 2, 1], [2, 1, 0.5]]
}

base_config = "hopper_traj_single_int"
metric_name = 'mean_reward'  # the metric we are tuning to optimize


def get_new_log_dir(old_dirs):
    """Get the directory of the newly completed run."""
    logs_dir = Path(LEGGED_GYM_ROOT_DIR) / 'logs' / base_config
    while True:
        current_dirs = set(os.listdir(logs_dir))
        new_dirs = current_dirs - old_dirs
        if new_dirs:
            return logs_dir / new_dirs.pop()
        time.sleep(1)


def get_metric_from_tensorboard(log_dir):
    """Read the metric from the TensorBoard logs."""
    event_file = next(log_dir.glob("events.out.tfevents.*"))
    event_acc = EventAccumulator(str(event_file))
    event_acc.Reload()
    if metric_name in event_acc.Tags()['scalars']:
        scalar_data = event_acc.Scalars(metric_name)
        return scalar_data[-1].value
    return None


def run_experiment(cfg, params, wandb_run):
    logs_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', base_config)

    for key, value in params.items():
        OmegaConf.update(cfg, key, value, merge=True)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]

    temp_config_file = "temp_config.yaml"
    temp_config_dir = Path("configs") / "rl"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_config_dir / temp_config_file

    with open(temp_config_path, 'w') as f:
        OmegaConf.save(cfg, f)

    # No need to reinitialize the Hydra config path
    old_dirs = set(os.listdir(logs_dir))
    train_rl_main(cfg)  # Assuming this function resets the environment for each run
    new_log_dir = get_new_log_dir(old_dirs)

    metric_value = get_metric_from_tensorboard(new_log_dir)
    if metric_value is not None:
        wandb_run.log({"mean_reward": metric_value, **flat_cfg_dict})

    return metric_value


def main():
    with hydra.initialize(config_path="configs/rl"):
        cfg = hydra.compose(config_name="hopper_trajectory_single_int")

    # Initialize WandB once for the entire run
    wandb_run = wandb.init(
        project="RL_Training",
        entity="coleonguard-Georgia Institute of Technology",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    for params in itertools.product(*param_grid.values()):
        params_dict = dict(zip(param_grid.keys(), params))
        print(f"Running with parameters: {params_dict}")
        result = run_experiment(cfg, params_dict, wandb_run)
        print(f"Result: {result}")

    # Finish the WandB run after all experiments
    wandb_run.finish()


if __name__ == "__main__":
    main()
