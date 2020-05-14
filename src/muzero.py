from collections import namedtuple

import torch
import torch.nn as nn
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple

from src.model_trainer import ValueNetwork, PolicyNetwork, RewardNetwork, TransitionModel, from_categorical, \
    inverse_transform, NetworkOutput
from src.rlpyt_agents import DQNSearchAgent, VectorizedMCTS
from src.rlpyt_models import RepNet

AgentInfo = namedarraytuple("AgentInfo", "p")
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
ModelSamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                       ["observation", "action", "reward", "done", "value"])
OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
ModelOptInfo = namedtuple("OptInfo", ["loss", "gradNorm",
                                      "tdAbsErr",
                                      "modelRLLoss",
                                      "RewardLoss",
                                      "modelGradNorm"])



class MuZeroAgent(DQNSearchAgent):
    def initialize(self,
                   env_spaces,
                   share_memory=False,
                   global_B=1,
                   env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B, env_ranks)
        # TODO: Handle epsilon greedy.
        self.search = VectorizedMCTS(self.search_args,
                                      env_spaces.action.n,
                                      self.model,
                                      eval=self.eval,
                                      distribution=self.distribution)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Compute the discrete distribution for the Q-value for each
        action for each state/observation (no grad)."""
        action, p, value, initial_value = self.search.run(observation.to(self.search.device))
        p = p.cpu()
        action = action.cpu()

        agent_info = AgentInfo(p=p)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)


class MuZeroModel(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            output_size,
            channels=None,  # None uses default.
            framestack=4,
            grayscale=True,
            actions=False,
            jumps=0,
            detach_model=True,
            stack_actions=False,
            dynamics_blocks=16,
            film=False,
            norm_type="bn",
            imagesize=84):
        # TODO: Rename output_size to num_actions (both here and in run_rlpyt.py)
        super().__init__()
        f, c, h, w = image_shape
        self.conv = RepNet(f*c, norm_type=norm_type)
        self.jumps = jumps
        self.stack_actions = stack_actions
        self.value_head = ValueNetwork(input_channels=256, hidden_channels=1)
        self.policy_head = PolicyNetwork(input_channels=256, num_actions=output_size, hidden_channels=2)
        self.dynamics_model = TransitionModel(256, output_size, blocks=dynamics_blocks, norm_type=norm_type)

    def forward(self, obs):
        pass

    def initial_inference(self, obs, actions=None, logits=False):
        if len(obs.shape) == 5:
            obs = obs.flatten(1, 2)
        hidden_state = self.conv(obs)
        policy_logits = self.policy_head(hidden_state)
        value_logits = self.value_head(hidden_state)
        reward_logits = self.dynamics_model.reward_predictor(hidden_state)

        if logits:
            return hidden_state, reward_logits, policy_logits, value_logits

        value = inverse_transform(from_categorical(value_logits, logits=True, limit=300))  #TODO Make these configurable
        reward = inverse_transform(from_categorical(reward_logits, logits=True, limit=300))
        return hidden_state, reward, policy_logits, value

    def inference(self, hidden_state, action):
        next_state, reward_logits, \
        policy_logits, value_logits = self.step(hidden_state, action)
        value = inverse_transform(from_categorical(value_logits,
                                                   logits=True))
        reward = inverse_transform(from_categorical(reward_logits,
                                                    logits=True))

        return NetworkOutput(next_state, reward, policy_logits, value)

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        policy_logits = self.policy_head(next_state)
        value_logits = self.value_model(next_state)

        return next_state, reward_logits, policy_logits, value_logits


class MuZeroAlgo(RlAlgorithm):
    def __init__(self, jumps=5, reward_loss_weight=1., policy_loss_weight=1., value_loss_weight=1.,
                 **kwargs):
        super().__init__(**kwargs)
        self.jumps = jumps
        self.opt_info_fields = tuple(f for f in ModelOptInfo._fields)  # copy
        self.reward_loss_weight = reward_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
