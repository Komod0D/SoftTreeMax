import gym
from gym import spaces
import numpy as np


class Failure(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_states: int = 3, time_horizon: int = 10):
        super().__init__()
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = self.observation_space
        self.current_state = self.observation_space.sample()
        self.time_horizon = time_horizon
        self.time_step = 0

    def step(self, action):
        observation = action
        reward = 1 if action == 0 and self.current_state == 0 else 0
        self.time_step += 1
        terminated = self.time_step == self.time_horizon

        truncated = False
        info = {"state": self.current_state, "action": action, "reward": reward}

        if terminated:
            observation, info = self.reset()

        self.current_state = action
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.current_state = self.observation_space.sample()
        observation = self.current_state
        
        info = {"reset": True}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass
