from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from src.mcts import MCTS
import torch

AgentInfo = namedarraytuple("AgentInfo", ["policy", "value"])
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])


class MCTSAgent(AtariDqnAgent):
    def __init__(self, search_args=None, eval=False, **kwargs):
        """Standard init, and set the number of probability atoms (bins)."""
        super().__init__(**kwargs)
        self.search_args = search_args
        self.eval = eval

    def initialize(self,
                   env_spaces,
                   share_memory=False,
                   global_B=1,
                   env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B, env_ranks)
        # Overwrite distribution.
        self.search = MCTS(self.search_args, env_spaces.action.n, self.model, eval=self.eval)

    def eval_mode(self, itr):
        """Extend method to set epsilon for evaluation, using 1 for
        pre-training eval."""
        super().eval_mode(itr)
        self.search.set_eval()
        self.search.epsilon = self.distribution.epsilon
        self.search.network.head.set_sampling(False)
        self.itr = itr

    def sample_mode(self, itr):
        """Extend method to set epsilon for sampling (including annealing)."""
        super().sample_mode(itr)
        self.search.epsilon = self.distribution.epsilon
        self.search.network.head.set_sampling(True)
        self.itr = itr

    def train_mode(self, itr):
        super().train_mode(itr)
        self.search.network.head.set_sampling(True)
        self.itr = itr

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Compute the discrete distribution for the Q-value for each
        action for each state/observation (no grad)."""
        action, pi_bar, value = self.search.run(observation.to(self.search.device))
        p = pi_bar.probs.cpu()
        action = action.cpu()

        agent_info = AgentInfo(policy=p, value=value.cpu())
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)
