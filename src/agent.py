import torch
from rlpyt.agents.dqn.atari.atari_catdqn_agent import AtariCatDqnAgent
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
AgentInputs = namedarraytuple("AgentInputs",
    ["observation", "prev_action", "prev_reward"])
AgentInfo = namedarraytuple("AgentInfo", "p")
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])


class SPRAgent(AtariCatDqnAgent):
    """Agent for Categorical DQN algorithm with search."""

    def __init__(self, eval=False, **kwargs):
        """Standard init, and set the number of probability atoms (bins)."""
        super().__init__(**kwargs)
        self.eval = eval

    def __call__(self, observation, prev_action, prev_reward, train=False):
        """Returns Q-values for states/observations (with grad)."""
        if train:
            model_inputs = buffer_to((observation, prev_action, prev_reward),
                device=self.device)
            return self.model(*model_inputs, train=train)
        else:
            prev_action = self.distribution.to_onehot(prev_action)
            model_inputs = buffer_to((observation, prev_action, prev_reward),
                device=self.device)
            return self.model(*model_inputs).cpu()

    def initialize(self,
                   env_spaces,
                   share_memory=False,
                   global_B=1,
                   env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B, env_ranks)
        # Overwrite distribution.
        self.search = SPRActionSelection(self.model, self.distribution)

    def to_device(self, cuda_idx=None):
        """Moves the model to the specified cuda device, if not ``None``.  If
        sharing memory, instantiates a new model to preserve the shared (CPU)
        model.  Agents with additional model components (beyond
        ``self.model``) for action-selection or for use during training should
        extend this method to move those to the device, as well.

        Typically called in the runner during startup.
        """
        super().to_device(cuda_idx)
        self.search.to_device(cuda_idx)
        self.search.network = self.model

    def eval_mode(self, itr):
        """Extend method to set epsilon for evaluation, using 1 for
        pre-training eval."""
        super().eval_mode(itr)
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
        action, p = self.search.run(observation.to(self.search.device))
        p = p.cpu()
        action = action.cpu()

        agent_info = AgentInfo(p=p)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)


class SPRActionSelection(torch.nn.Module):
    def __init__(self, network, distribution, device="cpu"):
        super().__init__()
        self.network = network
        self.epsilon = distribution._epsilon
        self.device = device
        self.first_call = True

    def to_device(self, idx):
        self.device = idx

    @torch.no_grad()
    def run(self, obs):
        while len(obs.shape) <= 4:
            obs.unsqueeze_(0)
        obs = obs.to(self.device).float() / 255.

        value = self.network.select_action(obs)
        action = self.select_action(value)
        # Stupid, stupid hack because rlpyt does _not_ handle batch_b=1 well.
        if self.first_call:
            action = action.squeeze()
            self.first_call = False
        return action, value.squeeze()

    def select_action(self, value):
        """Input can be shaped [T,B,Q] or [B,Q], and vector epsilon of length
        B will apply across the Batch dimension (same epsilon for all T)."""
        arg_select = torch.argmax(value, dim=-1)
        mask = torch.rand(arg_select.shape, device=value.device) < self.epsilon
        arg_rand = torch.randint(low=0, high=value.shape[-1], size=(mask.sum(),), device=value.device)
        arg_select[mask] = arg_rand
        return arg_select
