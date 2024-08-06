import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import hydra
import wandb
import torch
import pandas as pd
from pathlib import Path
from functools import partial
from omegaconf import OmegaConf
from deep_tube_learning.utils import update_cfgs_from_hydra, update_args_from_hydra, policy_runner_wandb_callback, CheckPointManager, wandb_load_artifact


@hydra.main(
    config_path=str(Path(__file__).parent / "configs" / "rl"),
    config_name="hopper_trajectory_single_int",
    version_base="1.2",
)
def main(cfg):
    # Send config to wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = pd.json_normalize(cfg_dict, sep="/").to_dict(orient="records")[0]
    wandb.init(project="RL_Training",
               entity="coleonguard-Georgia Institute of Technology",
               name=cfg.experiment_name,
               config=cfg_dict)

    args = get_args()
    args = update_args_from_hydra(cfg, args)
    env_cfg, train_cfg = task_registry.get_cfgs(cfg.task)
    env_cfg, train_cfg = update_cfgs_from_hydra(cfg, env_cfg, train_cfg)

    env, env_cfg = task_registry.make_env(name=cfg.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=cfg.task,
        args=args,
        train_cfg=train_cfg,
        wandb_callback=partial(
            policy_runner_wandb_callback,
            steps_per_model_checkpoint=cfg.steps_per_model_checkpoint,
            checkpointer=CheckPointManager(metric_name="mean_reward")
        )
    )
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

    for c in range(env.curriculum_state + 1):
        model_name = f'coleonguard-Georgia Institute of Technology/RL_Training/{wandb.run.id}_model:best{c}'
        api = wandb.Api()
        config, artifact = wandb_load_artifact(api, model_name)

        dir_name = artifact.download(root=Path(f"models/{wandb.run.id}/best_model"))
        ppo_runner.load(f'{dir_name}/model.pth')
        obs = env.get_observations()
        ppo_runner.alg.actor_critic.eval()
        onnx_path = f"models/{wandb.run.id}/{cfg.experiment_name}_cur{c}_{wandb.run.id}.onnx"
        torch.onnx.export(
            ppo_runner.alg.actor_critic.actor,
            obs[0].clone().detach(),
            onnx_path,
            export_params=True
        )

        print('Exported policy as onnx file to: ', onnx_path)
    print(f"Finishing run {wandb.run.id}")
    print("Waiting for wandb to finish uploading...")


if __name__ == "__main__":
    main()
