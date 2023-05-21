# Built-in
from typing import Tuple
import math

# Externals
import torch as th
from stable_baselines3.common.utils import get_device

# Internals
from policies.actor_critic_depth0 import ActorCriticPolicyDepth0
from policies.cule_bfs import CuleBFS, FailureBFS
from utils import add_regularization_logits


class ActorCriticTSPolicy(ActorCriticPolicyDepth0):
    def __init__(self, observation_space, action_space, lr_schedule, tree_depth, gamma, step_env, buffer_size,
                 learn_alpha, learn_beta, max_width, use_leaves_v, is_cumulative_mode, regularization, **kwargs):
        super(ActorCriticTSPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)        
        self.cule_bfs = FailureBFS(step_env, tree_depth, gamma, self.compute_value, max_width)
        self.time_step = 0
        self.obs2leaves_dict = {}
        self.timestep2obs_dict = {}
        self.obs2timestep_dict = {}
        self.buffer_size = buffer_size
        self.learn_alpha = False
        self.learn_beta = learn_beta
        self.tree_depth = tree_depth
        self.max_width = max_width
        self.is_cumulative_mode = is_cumulative_mode
        self.regularization = regularization
        self.alpha = th.tensor(0.5 if learn_alpha else 1.0, device=self.device)
        self.beta = th.tensor(1.0, device=self.device)
        self.lr_schedule = lr_schedule
        if self.learn_alpha:
            self.alpha = th.nn.Parameter(self.alpha)
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        if self.learn_beta:
            self.beta = th.nn.Parameter(self.beta)
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.use_leaves_v = use_leaves_v

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        
        mean_actions_logits = th.ones((self.action_space.n, )) / self.action_space.n
        mean_actions_logits[0] += 1
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        self.time_step += 1
        return actions, 0, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        self.add_gradients_history()
        batch_size = obs.shape[0]
        
        mean_actions_logits = th.ones((batch_size, self.action_space.n)) / self.action_space.n
        mean_actions_logits[:, 0] += 1

        distribution = self.action_dist.proba_distribution(action_logits=mean_actions_logits)
        log_prob = distribution.log_prob(actions)
        return 0, log_prob, distribution.entropy()

    def hash_obs(self, obs):
        return obs

    def compute_value_with_root(self, leaves_obs, root_obs=None):
        if root_obs is None:
            shared_features = self.mlp_extractor.shared_net(self.extract_features(leaves_obs))
            return self.action_net(self.mlp_extractor.policy_net(shared_features)), None
        cat_features = self.extract_features(th.cat((root_obs.reshape(-1, 1), leaves_obs)))
        shared_features = self.mlp_extractor.shared_net(cat_features)
        latent_pi = self.mlp_extractor.policy_net(shared_features[1:])
        latent_vf_root = self.mlp_extractor.value_net(shared_features[:1])
        value_root = self.value_net(latent_vf_root)
        return latent_pi, value_root

    def compute_value(self, leaves_obs, root_obs=None):
        if root_obs is None:
            shared_features = self.mlp_extractor.shared_net(self.extract_features(leaves_obs))
            return self.action_net(self.mlp_extractor.policy_net(shared_features)), None
        shared_features = self.mlp_extractor.shared_net(self.extract_features(leaves_obs))
        latent_pi = self.mlp_extractor.policy_net(shared_features)
        latent_vf_root = self.mlp_extractor.value_net(shared_features)
        value_root = self.value_net(latent_vf_root)
        return latent_pi, value_root

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path: (str)
        """
        time_step = self.time_step,
        th.save({"state_dict": self.state_dict(), "data": self._get_data(),
                 "alpha": self.alpha, "beta": self.beta, "time_step": self.time_step}, path)

    def _get_data(self):
        """
        Get data that need to be saved in order to re-create the model.
        This corresponds to the arguments of the constructor.

        :return: (Dict[str, Any])
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            tree_depth=self.tree_depth,
            gamma=self.cule_bfs.gamma,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            buffer_size=self.buffer_size,
            learn_alpha=self.learn_alpha,
            learn_beta=self.learn_beta,
            is_cumulative_mode=self.is_cumulative_mode,
            regularization=self.regularization,
            max_width=self.max_width,
            use_leaves_v=self.use_leaves_v,
        )

    @classmethod
    def load(cls, path, device="auto", env=None, lr_schedule=None):
        """
        Load model from path.

        :param path: (str)
        :param device: (Union[th.device, str]) Device on which the policy should be loaded.
        :return: (BasePolicy)
        """
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables["data"], lr_schedule=lr_schedule, step_env=env)  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def predict(self, obs: th.Tensor, state=None, mask=None, deterministic: bool = False):
        return self.forward(th.tensor(obs, dtype=th.float32, device=get_device()), deterministic)[0], None
