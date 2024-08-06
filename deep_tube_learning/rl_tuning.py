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

# Define the grid of hyperparameters to tune
param_grid = {
    'env_config.curriculum.rewards.tracking_rom': [[1.0, 0.8, 0.6], [1.2, 1.0, 0.8]],
    'env_config.curriculum.push.magnitude': [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]],
    'env_config.curriculum.push.time': [[3, 2, 1], [2, 1, 0.5]]
}

logs_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'hopper_traj_single_int')
metric_name = 'mean_reward'  # the metric we are tuning to optimize


def get_new_log_dir(old_dirs):
    """Get the directory of the newly completed run."""
    while True:
        current_dirs = set(os.listdir(logs_dir))
        new_dirs = current_dirs - old_dirs
        if new_dirs:
            return new_dirs.pop()
        time.sleep(1)  # Wait a bit before checking again


def get_metric_from_tensorboard(log_dir):
    """Read the metric from the TensorBoard logs."""
    event_file = next(log_dir.glob("events.out.tfevents.*"))
    event_acc = EventAccumulator(str(event_file))
    event_acc.Reload()

    # Assuming we are interested in a specific scalar metric, e.g., 'mean_reward'
    if metric_name in event_acc.Tags()['scalars']:
        scalar_data = event_acc.Scalars(metric_name)
        return scalar_data[-1].value  # Return the last logged value for this metric
    return None


def run_experiment(cfg, params):
    # Update the Hydra config with the current set of parameters
    for key, value in params.items():
        OmegaConf.update(cfg, key, value, merge=True)

    # Convert the updated config to a flat dictionary for WandB logging
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]

    # Initialize WandB run
    wandb.init(project="RL_Training",
               entity="coleonguard-Georgia Institute of Technology",
               name=cfg.experiment_name,
               config=flat_cfg_dict)

    # Set the directory and file name for the temporary config file
    temp_config_file = "temp_config.yaml"
    temp_config_dir = Path(__file__).parent / "configs" / "rl"

    # Ensure the directory exists
    temp_config_dir.mkdir(parents=True, exist_ok=True)

    # Save the updated config to the temporary file
    temp_config_path = temp_config_dir / temp_config_file
    # with open(temp_config_path, 'w') as f:
    #     OmegaConf.save(cfg, f)

    # Get the current set of directories to detect new runs
    old_dirs = set(os.listdir(logs_dir))

    # Construct the command to run the training script
    command = ["python", str(Path(__file__).parent / "train_rl.py"), "-cn=" + temp_config_path.stem]

    print("Running command:", " ".join(command))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Output: {result.stdout}")

    # Get the directory of the newly completed run
    new_log_dir = get_new_log_dir(old_dirs)

    # Parse the metric from the TensorBoard logs
    metric_value = get_metric_from_tensorboard(new_log_dir)

    # Log the result along with the configuration to WandB
    if metric_value is not None:
        wandb.log({"mean_reward": metric_value, **flat_cfg_dict})

    # Finish the WandB run
    wandb.finish()

    return metric_value


def main():
    # Load your base Hydra configuration
    with hydra.initialize(config_path="configs/rl"):
        cfg = hydra.compose(config_name="hopper_trajectory_single_int")

    # Iterate over all combinations of hyperparameters
    for params in itertools.product(*param_grid.values()):
        params_dict = dict(zip(param_grid.keys(), params))
        print(f"Running with parameters: {params_dict}")

        # Run the experiment and get the result
        result = run_experiment(cfg, params_dict)

        print(f"Result: {result}")


if __name__ == "__main__":
    main()
