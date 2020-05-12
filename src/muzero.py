import torch
import torch.nn as nn
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent

from src.model_trainer import ValueNetwork, PolicyNetwork, RewardNetwork, TransitionModel
from src.rlpyt_agents import DQNSearchAgent, VectorizedMCTS
from src.rlpyt_models import RepNet


class MuZeroAgent(DQNSearchAgent):
    def initialize(self,
                   env_spaces,
                   share_memory=False,
                   global_B=1,
                   env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B, env_ranks)
        # Overwrite distribution.
        self.search = VectorizedMCTS(self.search_args,
                                      env_spaces.action.n,
                                      self.model,
                                      eval=self.eval,
                                      distribution=self.distribution)


class MuZeroModel(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,
            dueling=False,
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
            encoder="repnet",
            noisy_nets=0,
            imagesize=84):
        super().__init__()
        self.dueling = dueling
        f, c, h, w = image_shape
        self.conv = RepNet(f*c, norm_type=norm_type)
        self.jumps = jumps
        self.stack_actions = stack_actions
        self.value_head = ValueNetwork()
        self.policy_head = PolicyNetwork()
        self.reward_head = RewardNetwork()
        self.dynamics_model = TransitionModel()



class MuZeroAlgo():
    pass