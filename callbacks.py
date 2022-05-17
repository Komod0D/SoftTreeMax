import numpy as np
import wandb
import os
import sys

from stable_baselines3.common.callbacks import BaseCallback


class WandbTrainingCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(WandbTrainingCallback, self).__init__(verbose)
        self.total_rewards = 0
        self.episode_length = 0
        self.prev_life = 0

    def _on_step(self) -> bool:
        # self.total_rewards += np.mean(self.locals["rewards"])
        infos = self.locals["infos"]
        if "orig_reward" in infos:
            self.total_rewards += np.mean(infos["orig_reward"])
        elif type(infos) == list:
            self.total_rewards += np.mean([info["orig_reward"] for info in infos])
        self.episode_length += 1

        # if "ale.lives" in info:
        #     ale_lives = int(info["ale.lives"][0]) if type(info["ale.lives"]) == list else int(info["ale.lives"])
        #     done = (ale_lives > self.prev_life)
        #     if done:
        #         if self.prev_life <= 1:
        #             print("Error: Lives went up but did not reach 0 before!")
        #         # assert self.prev_life <= 1, "Error: Lives went up but did not reach 0 before!"
        #     self.prev_life = ale_lives
        if "done" in infos:
            done = infos["done"]
        else:
            done = int(self.locals["dones"][0])
        if type(done) == np.ndarray:
            done = any(done)
        # if False:
        if done and sys.gettrace() is None:
            # TODO: fix here - depth 0 has no alpha
            if hasattr(self.locals["self"].policy, "alpha"):
                wandb.log({"train\\alpha": self.locals["self"].policy.alpha.item()}, step=self.model.num_timesteps)
            wandb.log({"train\episodic_reward": self.total_rewards}, step=self.model.num_timesteps)
            wandb.log({"train\episodic_length": self.episode_length}, step=self.model.num_timesteps)
            wandb.log({"num_steps": self.model.num_timesteps}, step=self.model.num_timesteps)
            if hasattr(self.locals["self"].policy, "cule_bfs"):
                wandb.log({"effective depth": np.mean(self.locals["self"].policy.cule_bfs.effective_depth)}, step=self.model.num_timesteps)
                self.locals["self"].policy.cule_bfs.effective_depth = []
                wandb.log({"train\\gradient_norm": np.mean(self.locals["self"].policy.gradient_norm)},
                          step=self.model.num_timesteps)
                self.locals["self"].policy.gradient_norm = []
            for key, val in self.locals["self"].logger.name_to_value.items():
                wandb.log({key: val}, step=self.model.num_timesteps)
            self.total_rewards = 0
            self.episode_length = 0

