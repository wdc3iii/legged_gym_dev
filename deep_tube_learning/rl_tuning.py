import hydra
from omegaconf import OmegaConf
import wandb
import itertools
import subprocess
import os
import time
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from legged_gym import LEGGED_GYM_ROOT_DIR
from train_rl import main as train_rl_main

# Define the grid of hyperparameters to tune
param_grid = {
    'env_config.rewards.scales.termination': [-1000,
                                              -500],

    'env_config.rewards.scales.action_rate': [-0.1,
                                              -0.01],

    'env_config.curriculum.rewards.termination': [[1.0, 0.5, 0.2, 0.1],
                                                  [1.5, 1.0, 0.9, 0.8]],

    'env_config.curriculum.rewards.collision': [[1.0, 0.5, 0.2, 0.1],
                                                [1.5, 1.0, 0.9, 0.8]],

    'env_config.curriculum.rewards.torques': [[.5, 0.6, 0.8, 1.0],
                                              [1.0, .8, 0.5, 0.2]],

    'env_config.curriculum.rewards.action_rate': [[.5, 0.6, 0.8, 1.0],
                                                  [1.0, .8, 0.5, 0.2]],

    'env_config.curriculum.sigma.tracking_rom': [[1.0, 0.9, 0.8, 0.6],
                                                 [1.0, 0.5, 0.2, 0.1]],  # we're tuning the rest for this one (how low can sigma get with good performance)
}

base_config = "hopper_traj_single_int"
metric_name = 'Episode/rew_tracking_rom'  # the metric we are tuning to optimize
last_num_values = 5  # the number of values at the end of the run to average over for the metric value
log_file_path = "experiment_log.json"


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
    """Read the metric from the TensorBoard logs and calculate the mean of the last 5 episodes."""
    event_file = next(log_dir.glob("events.out.tfevents.*"))
    event_acc = EventAccumulator(str(event_file))
    event_acc.Reload()

    if metric_name in event_acc.Tags()['scalars']:
        scalar_data = event_acc.Scalars(metric_name)
        last_x_values = [scalar.value for scalar in scalar_data[-last_num_values:]]
        return sum(last_x_values) / len(last_x_values) if last_x_values else None
    return None


def save_log(log_data):
    """Save the experiment data to a JSON log file."""
    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)


def plot_metrics(log_data):
    """Plot the metric values for each run."""
    run_numbers = list(range(1, len(log_data) + 1))
    metric_values = [entry['metric_value'] for entry in log_data]

    plt.figure(figsize=(10, 6))
    plt.bar(run_numbers, metric_values, color='skyblue')
    plt.xlabel('Run Number')
    plt.ylabel('Mean Reward')
    plt.title(f'Mean {metric_name}')
    plt.xticks(run_numbers)
    plt.savefig('experiment_results.png')
    plt.show()


def run_experiment(cfg, params, wandb_run):
    for key, value in params.items():
        OmegaConf.update(cfg, key, value, merge=True)

        if key == 'env_config.trajectory_generator.N':
            num_observations = 18 + value
            OmegaConf.update(cfg, 'env_config.env.num_observations', num_observations, merge=True)


    logs_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', base_config)
    temp_config_file = "temp_config.yaml"
    temp_config_dir = Path("configs") / "rl"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_config_dir / temp_config_file

    with open(temp_config_path, 'w') as f:
        OmegaConf.save(cfg, f)

    old_dirs = set(os.listdir(logs_dir))
    train_rl_main(cfg)
    new_log_dir = get_new_log_dir(old_dirs)
    metric_value = get_metric_from_tensorboard(new_log_dir)

    if metric_value is not None:
        wandb_run.log({"mean_reward": metric_value, **params})

    return metric_value, params


def main():
    with hydra.initialize(config_path="configs/rl"):
        cfg = hydra.compose(config_name="hopper_trajectory_single_int")

    wandb_run = wandb.init(
        project="RL_Training",
        entity="coleonguard-Georgia Institute of Technology",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    log_data = []

    for run_number, params in enumerate(itertools.product(*param_grid.values()), start=1):
        params_dict = dict(zip(param_grid.keys(), params))
        print(f"Running with parameters: {params_dict}")
        metric_value, config_used = run_experiment(cfg, params_dict, wandb_run)
        print(f"Result: {metric_value}")

        log_entry = {
            "run_number": run_number,
            "config": config_used,
            "metric_value": metric_value
        }
        log_data.append(log_entry)

    save_log(log_data)
    plot_metrics(log_data)
    wandb_run.finish()


if __name__ == "__main__":
    main()
