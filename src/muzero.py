from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.agents.dqn.atari.mixin import AtariMixin
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from torch.nn.parallel import DistributedDataParallel as DDP

from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args

from src.mcts_memory import AsyncPrioritizedSequenceReplayFrameBufferExtended, \
    AsyncUniformSequenceReplayFrameBufferExtended
from src.model_trainer import ValueNetwork, PolicyNetwork, RewardNetwork, TransitionModel, from_categorical, \
    inverse_transform, NetworkOutput, to_categorical, transform
from src.rlpyt_agents import DQNSearchAgent, VectorizedMCTS
from src.rlpyt_models import RepNet

AgentInfo = namedarraytuple("AgentInfo", ["policy_probs", "value"])
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])
ModelSamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                       ["observation", "action", "reward", "done", "policy_probs", "value"])
OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
ModelOptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr",
                                      "RewardLoss", "PolicyLoss", "ValueLoss"])


class MuZeroAgent(AtariMixin, DqnAgent):
    def __init__(self, search_args=None, eval=False, **kwargs):
        """Standard init"""
        super().__init__(**kwargs)
        self.search_args = search_args
        self.eval = eval

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

        agent_info = AgentInfo(policy_probs=p, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

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
        self.search.set_eval()

    def sample_mode(self, itr):
        """Extend method to set epsilon for sampling (including annealing)."""
        super().sample_mode(itr)
        self.search.set_train()

    def data_parallel(self):
        """Wraps the model with PyTorch's DistributedDataParallel.  The
        intention is for rlpyt to create a separate Python process to drive
        each GPU (or CPU-group for CPU-only, MPI-like configuration). Agents
        with additional model components (beyond ``self.model``) which will
        have gradients computed through them should extend this method to wrap
        those, as well.

        Typically called in the runner during startup.
        """
        self.model = DDP(self.model,
                         device_ids=[self.device.index],
                         output_device=self.device.index,
                         broadcast_buffers=False,
                         find_unused_parameters=True)
        logger.log("Initialized DistributedDataParallel agent model on "
                   f"device {self.device}.")


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
        self.hidden_size = 256
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
        value_logits = self.value_head(next_state)

        return next_state, reward_logits, policy_logits, value_logits


class MuZeroAlgo(RlAlgorithm):
    def __init__(self, jumps=5, reward_loss_weight=1., policy_loss_weight=1., value_loss_weight=1.,
                 discount=0.99,
                 batch_size=32,
                 min_steps_learn=int(5e4),
                 replay_size=int(1e6),
                 replay_ratio=8,  # data_consumption / data_generation.
                 target_update_tau=1,
                 target_update_interval=312,  # 312 * 32 = 1e4 env steps.
                 n_step_return=1,
                 learning_rate=2.5e-4,
                 OptimCls=torch.optim.Adam,
                 optim_kwargs=None,
                 initial_optim_state_dict=None,
                 clip_grad_norm=10.,
                 prioritized_replay=False,
                 pri_alpha=1.,
                 pri_beta_init=1.,
                 pri_beta_final=1.,
                 pri_beta_steps=int(50e6),
                 default_priority=None,
                 ReplayBufferCls=None,  # Leave None to select by above options.
                 updates_per_sync=1,  # For async mode only.
                 use_target_network=True,
                 **kwargs):
        self.jumps = jumps
        self.updates_per_sync = updates_per_sync
        self.opt_info_fields = tuple(f for f in ModelOptInfo._fields)  # copy
        self.reward_loss_weight = reward_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())
        self.update_counter = 0

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        example_to_buffer = ModelSamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            policy_probs=examples["agent_info"].policy_probs,
            value=examples["agent_info"].value
        )
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            batch_T=self.jumps+1,
            discount=self.discount,
            n_step_return=self.n_step_return,
            rnn_state_interval=0,
        )

        if self.prioritized_replay:
            replay_kwargs['alpha'] = self.pri_alpha
            replay_kwargs['beta'] = self.pri_beta_init
            # replay_kwargs["input_priorities"] = self.input_priorities
            buffer = AsyncPrioritizedSequenceReplayFrameBufferExtended(**replay_kwargs)
        else:
            buffer = AsyncUniformSequenceReplayFrameBufferExtended(**replay_kwargs)

        self.replay_buffer = buffer

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
                   world_size=1, rank=0):
        """Stores input arguments and initializes replay buffer and optimizer.
        Use in non-async runners.  Computes number of gradient updates per
        optimization iteration as `(replay_ratio * sampler-batch-size /
        training-batch_size)`."""
        self.agent = agent
        self.n_itr = n_itr
        self.sampler_bs = sampler_bs = batch_spec.size
        self.mid_batch_reset = mid_batch_reset
        self.updates_per_optimize = max(1, round(self.replay_ratio * sampler_bs /
                                                 self.batch_size))
        logger.log(f"From sampler batch size {batch_spec.size}, training "
                   f"batch size {self.batch_size}, and replay ratio "
                   f"{self.replay_ratio}, computed {self.updates_per_optimize} "
                   f"updates per iteration.")
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        self.initialize_replay_buffer(examples, batch_spec)
        self.optim_initialize(rank)

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset,
                         examples, world_size=1):
        """Used in async runner only; returns replay buffer allocated in shared
        memory, does not instantiate optimizer. """
        self.agent = agent
        self.n_itr = sampler_n_itr
        self.initialize_replay_buffer(examples, batch_spec, async_=True)
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = self.updates_per_sync
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        return self.replay_buffer

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        try:
            # We're probably dealing with DDP
            self.optimizer = self.OptimCls(self.agent.model.module.parameters(),
                                           lr=self.learning_rate, **self.optim_kwargs)
            self.model = self.agent.model.module
        except:
            self.optimizer = self.OptimCls(self.agent.model.parameters(),
                                           lr=self.learning_rate, **self.optim_kwargs)
            self.model = self.agent.model
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        if self.prioritized_replay:
            self.pri_beta_itr = max(1, self.pri_beta_steps // self.sampler_bs)

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method.  In
        asynchronous mode, will be called in the memory_copier process."""
        return ModelSamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
            policy_probs=samples.agent.agent_info.policy_probs,
            value=samples.agent.agent_info.value
        )

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Extracts the needed fields from input samples and stores them in the
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).  If using prioritized
        replay, updates the priorities for sampled training batches.
        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.=
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = ModelOptInfo(*([] for _ in range(len(ModelOptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            loss, td_abs_errors, reward_loss, policy_loss, value_loss = self.loss(samples_from_replay)
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(td_abs_errors)
            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(torch.tensor(grad_norm).item())  # grad_norm is a float sometimes, so wrap in tensor
            opt_info.RewardLoss.append(reward_loss.item())
            opt_info.PolicyLoss.append(policy_loss.item())
            opt_info.ValueLoss.append(value_loss.item())
            opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.
            self.update_counter += 1
            if self.update_counter % self.target_update_interval and self.use_target_network == 0:
                self.agent.update_target(self.target_update_tau)
        return opt_info

    def loss(self, samples):
        obs = samples.all_observation[0].to(self.agent.device).float() / 255.
        actions = samples.all_action.long().to(self.agent.device)
        rewards = samples.all_reward.float().to(self.agent.device)
        policies = samples.policy_probs.float().to(self.agent.device)
        current_state, pred_reward, pred_policy, pred_value = self.model.initial_inference(obs,
                                                                                           logits=True)
        predictions = [(1.0, pred_reward, pred_policy, pred_value)]

        for i in range(1, self.jumps+1):
            action = actions[i]  # actions[i] is the action at step [i-1]
            current_state, pred_reward, pred_policy, pred_value = self.model.step(current_state, action)
            # TODO: Missing scale grad here
            # current_state = ScaleGradient.apply(current_state, self.args.grad_scale_factor)
            predictions.append((1., pred_reward, pred_policy, pred_value))

        reward_loss, policy_loss, value_loss = 0, 0, 0
        for i, prediction in enumerate(predictions):
            loss_scale, pred_reward, pred_policy, pred_value = prediction

            target_value = samples.return_[i] + \
                               (self.discount ** self.n_step_return * samples.values[i+self.n_step_return] *
                                (1 - samples.done_n[i].float()))
            target_value = to_categorical(transform(target_value)).to(self.agent.device)
            target_reward = to_categorical(transform(rewards[i]))

            pred_logvalue = F.log_softmax(pred_value, -1)
            pred_logreward = F.log_softmax(pred_reward, -1)
            pred_logpolicy = F.log_softmax(pred_policy, -1)

            current_reward_loss = (-target_reward * pred_logreward).sum(1)
            current_value_loss = (-target_value * pred_logvalue).sum(1)
            current_policy_loss = (-policies[i] * pred_logpolicy).sum(1)

            reward_loss += current_reward_loss
            value_loss += current_value_loss
            policy_loss += current_policy_loss

            if i == 0:
                pred_value = inverse_transform(from_categorical(pred_value.detach(), logits=True))
                target_value = inverse_transform(from_categorical(target_value.detach(), logits=True))
                value_error = torch.abs(pred_value - target_value).cpu()

        loss = self.value_loss_weight * value_loss + self.policy_loss_weight * policy_loss \
               + self.reward_loss_weight * reward_loss
        loss *= samples.is_weights[0]
        loss = loss.mean()
        return loss, value_error, reward_loss.mean(), policy_loss.mean(), value_loss.mean()

