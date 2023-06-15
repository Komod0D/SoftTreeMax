import gym
from gym import spaces
import numpy as np


def get_env(env_kwargs):
    env = env_kwargs['env_name']
    if env == 'Failure':
        return Failure(env_kwargs=env_kwargs)
    elif env == 'Step':
        return Step(env_kwargs=env_kwargs)
    elif env == 'Zigzag':
        return Zigzag(env_kwargs=env_kwargs)
    else:
        raise ValueError(f'"{env}" is not a correct environment name')
    
    
    
class Zigzag(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_states: int = 3, time_horizon: int = 10, env_kwargs=None):
        super().__init__()
        self.n_states = n_states
        self.env_kwargs = dict(env_name="Failure", n_states=n_states, time_horizon=time_horizon, env_kwargs=env_kwargs)
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(2)
        self.action_set = self.action_space
        self.current_state = self.observation_space.sample()
        self.time_horizon = time_horizon
        self.time_step = 0
        self.clip_reward = False
        self.frameskip = 0

    def step(self, action):
        direction = action * 2 - 1
        if self.current_state % 2 != 0:
            direction *= -1

        new_obs = self.current_state + direction
        observation = min(max(new_obs, 0), self.observation_space.n - 1)
        reward = 1 if observation == 0 and self.current_state == 0 else 0
        self.time_step += 1
        terminated = self.time_step >= self.time_horizon

        info = {"state": self.current_state, "action": action, "orig_reward": reward, "dones": True, "done": terminated, "reset": terminated}

        if terminated:
            observation = self.reset()

        self.current_state = observation
        return observation, reward, terminated, info

    def reset(self, seed=None, options=None, initial_steps=None, verbose=None):
        self.current_state = self.observation_space.sample()
        observation = self.current_state
        self.time_step = 0
        
        info = {"reset": True}
        return observation

    def render(self):
        pass

    def close(self):
        pass

    def copy(self):
        env = Zigzag(self.n_states, self.time_horizon, self.env_kwargs)
        env.current_state = self.current_state
        env.time_step = self.time_step

        return env

class Step(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_states: int = 3, time_horizon: int = 10, env_kwargs=None):
        super().__init__()
        self.n_states = n_states
        self.env_kwargs = dict(env_name="Failure", n_states=n_states, time_horizon=time_horizon, env_kwargs=env_kwargs)
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(2)
        self.action_set = self.action_space
        self.current_state = self.observation_space.sample()
        self.time_horizon = time_horizon
        self.time_step = 0
        self.clip_reward = False
        self.frameskip = 0

    def step(self, action):
        direction = action * 2 - 1
        new_obs = self.current_state + direction
        observation = min(max(new_obs, 0), self.observation_space.n - 1)
        reward = 1 if action == 0 and self.current_state == 0 else 0
        self.time_step += 1
        terminated = self.time_step >= self.time_horizon

        info = {"state": self.current_state, "action": action, "orig_reward": reward, "dones": True, "done": terminated, "reset": terminated}

        if terminated:
            observation = self.reset()

        self.current_state = observation
        return observation, reward, terminated, info

    def reset(self, seed=None, options=None, initial_steps=None, verbose=None):
        self.current_state = self.observation_space.sample()
        observation = self.current_state
        self.time_step = 0
        
        info = {"reset": True}
        return observation

    def render(self):
        pass

    def close(self):
        pass

    def copy(self):
        env = Step(self.n_states, self.time_horizon, self.env_kwargs)
        env.current_state = self.current_state
        env.time_step = self.time_step

        return env

class Failure(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_states: int = 3, time_horizon: int = 10, env_kwargs=None):
        super().__init__()
        self.n_states = n_states
        self.env_kwargs = dict(env_name="Failure", n_states=n_states, time_horizon=time_horizon, env_kwargs=env_kwargs)
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = self.observation_space
        self.action_set = self.action_space
        self.current_state = self.observation_space.sample()
        self.time_horizon = time_horizon
        self.time_step = 0
        self.clip_reward = False
        self.frameskip = 0

    def step(self, action):
        observation = action
        reward = 1 if action == 0 and self.current_state == 0 else 0
        self.time_step += 1
        terminated = self.time_step >= self.time_horizon

        info = {"state": self.current_state, "action": action, "orig_reward": reward, "dones": True, "done": terminated, "reset": terminated}

        if terminated:
            observation = self.reset()

        self.current_state = action
        return observation, reward, terminated, info

    def reset(self, seed=None, options=None, initial_steps=None, verbose=None):
        self.current_state = self.observation_space.sample()
        observation = self.current_state
        self.time_step = 0
        
        info = {"reset": True}
        return observation

    def render(self):
        pass

    def close(self):
        pass

    def copy(self):
        env = Failure(self.n_states, self.time_horizon, self.env_kwargs)
        env.current_state = self.current_state
        env.time_step = self.time_step

        return env