# -*- coding: utf-8 -*-
from __future__ import division
from collections import namedtuple
import numpy as np
import torch
from recordclass import recordclass

from rlpyt.replays.sequence.prioritized import SamplesFromReplayPri
from src.envs import get_example_outputs

from rlpyt.replays.non_sequence.frame import AsyncPrioritizedReplayFrameBuffer
from rlpyt.replays.sequence.n_step import SamplesFromReplay
from rlpyt.replays.sequence.frame import AsyncPrioritizedSequenceReplayFrameBuffer, \
    AsyncUniformSequenceReplayFrameBuffer, PrioritizedSequenceReplayFrameBuffer
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import extract_sequences
from rlpyt.utils.synchronize import RWLock

Transition = recordclass('Transition', ('timestep', 'state', 'action', 'reward', 'value', 'policy', 'nonterminal'))
blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), 0, 0., 0., 0, False)  # TODO: Set appropriate default policy value
blank_batch_trans = Transition(0, torch.zeros(1, 84, 84, dtype=torch.uint8), 0, 0., 0., 0, False)

PrioritizedSamples = namedarraytuple("PrioritizedSamples",
                                  ["samples", "priorities"])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done", "policy_probs", "value"])
SamplesFromReplayExt = namedarraytuple("SamplesFromReplayPriExt",
                                       SamplesFromReplay._fields + ("policy_probs", "values"))
SamplesFromReplayPriExt = namedarraytuple("SamplesFromReplayPriExt",
                                       SamplesFromReplayPri._fields + ("policy_probs", "values"))
EPS = 1e-6


def initialize_replay_buffer(args):
    examples = get_example_outputs(args)
    batch_size = args.num_envs
    if args.reanalyze:
        batch_size = batch_size + args.num_reanalyze_envs
    example_to_buffer = SamplesToBuffer(
        observation=examples["observation"],
        action=examples["action"],
        reward=examples["reward"],
        done=examples["done"],
        policy_probs=examples['policy_probs'],
        value=examples['value']
    )
    replay_kwargs = dict(
        example=example_to_buffer,
        size=args.buffer_size,
        B=batch_size,
        batch_T=args.jumps + args.multistep + 1,
        # We don't use the built-in n-step returns, so easiest to just ask for all the data at once.
        rnn_state_interval=0,
        discount=args.discount,
        n_step_return=1,
    )

    if args.prioritized:
        replay_kwargs["input_priorities"]=args.input_priorities,
        buffer = AsyncPrioritizedSequenceReplayFrameBufferExtended(**replay_kwargs)
    else:
        buffer = AsyncUniformSequenceReplayFrameBufferExtended(**replay_kwargs)

    return buffer


def samples_to_buffer(observation, action, reward, done, policy_probs, value, priorities=None):
    samples = SamplesToBuffer(
        observation=observation,
        action=action,
        reward=reward,
        done=done,
        policy_probs=policy_probs,
        value=value
        )
    if priorities is not None:
        return PrioritizedSamples(samples=samples,
                                  priorities=priorities)
    else:
        return samples

class AsyncUniformSequenceReplayFrameBufferExtended(AsyncUniformSequenceReplayFrameBuffer):
    """
    Extends AsyncPrioritizedSequenceReplayFrameBuffer to return policy_logits and values too during sampling.
    """
    def sample_batch(self, batch_B):
        self._async_pull()  # Updates from writers.
        batch_T = self.batch_T
        T_idxs, B_idxs = self.sample_idxs(batch_B, batch_T)
        # (T_idxs, B_idxs), priorities = self.priority_tree.sample(
        #     batch_B, unique=self.unique)
        if self.rnn_state_interval > 1:
            T_idxs = T_idxs * self.rnn_state_interval

        batch = self.extract_batch(T_idxs, B_idxs, self.batch_T)
        # is_weights = (1. / priorities) ** self.beta
        # is_weights /= max(is_weights)  # Normalize.
        # is_weights = torchify_buffer(is_weights).float()

        policy_probs = extract_sequences(self.samples.policy_probs, T_idxs, B_idxs, self.batch_T + self.n_step_return)
        values = extract_sequences(self.samples.value, T_idxs, B_idxs, self.batch_T + self.n_step_return)
        batch = SamplesFromReplayExt(*batch, policy_probs=policy_probs, values=values)
        return self.sanitize_batch(batch)

    def sanitize_batch(self, batch):
        has_dones, inds = torch.max(batch.done, 0)
        for i, (has_done, ind) in enumerate(zip(has_dones, inds)):
            if not has_done:
                continue
            batch.all_observation[ind+1:, i] = batch.all_observation[ind, i]
            batch.all_action[ind+1:, i] = batch.all_action[ind, i]
            batch.all_action[ind+1:, i] = batch.all_action[ind, i]
            batch.policy_probs[ind+1:, i] = batch.policy_probs[ind, i]
            batch.all_reward[ind+1:, i] = 0
            batch.values[ind+1:, i] = 0
        return batch


class AsyncPrioritizedSequenceReplayFrameBufferExtended(AsyncPrioritizedSequenceReplayFrameBuffer):
    """
    Extends AsyncPrioritizedSequenceReplayFrameBuffer to return policy_logits and values too during sampling.
    """
    def sample_batch(self, batch_B):
        self._async_pull()  # Updates from writers.
        batch_T = self.batch_T
        (T_idxs, B_idxs), priorities = self.priority_tree.sample(
            batch_B, unique=self.unique)
        if self.rnn_state_interval > 1:
            T_idxs = T_idxs * self.rnn_state_interval

        batch = self.extract_batch(T_idxs, B_idxs, self.batch_T)
        is_weights = (1. / priorities) ** self.beta
        is_weights /= max(is_weights)  # Normalize.
        is_weights = torchify_buffer(is_weights).float()

        policy_probs = extract_sequences(self.samples.policy_probs, T_idxs, B_idxs, self.batch_T + self.n_step_return)
        values = extract_sequences(self.samples.value, T_idxs, B_idxs, self.batch_T + self.n_step_return)
        batch = SamplesFromReplayPriExt(*batch, is_weights=is_weights, policy_probs=policy_probs, values=values)
        return self.sanitize_batch(batch)

    def update_batch_priorities(self, priorities):
        with self.rw_lock.write_lock:
            priorities = numpify_buffer(priorities)
            self.default_priority = max(priorities)
            self.priority_tree.update_batch_priorities(priorities ** self.alpha)

    def sanitize_batch(self, batch):
        has_dones, inds = torch.max(batch.done, 0)
        for i, (has_done, ind) in enumerate(zip(has_dones, inds)):
            if not has_done:
                continue
            batch.all_observation[ind+1:, i] = batch.all_observation[ind, i]
            batch.all_action[ind+1:, i] = batch.all_action[ind, i]
            batch.all_action[ind+1:, i] = batch.all_action[ind, i]
            batch.policy_probs[ind+1:, i] = batch.policy_probs[ind, i]
            batch.all_reward[ind+1:, i] = 0
            batch.values[ind+1:, i] = 0
        return batch


class LocalBuffer:
    """
    Helper class to store locally a single [num_timesteps (T), num_envs (B)] segment
    """
    def __init__(self, args):
        self.observations, self.actions, self.rewards, self.dones = [], [], [], []
        self.policy_logits, self.values, self.value_estimates = [], [], []
        self.args = args

    def clear(self):
        if self.args.input_priorities:
            self.observations, self.actions, self.rewards, self.dones, \
            self.policy_logits, self.values, self.value_estimates \
                = (t[-self.args.multistep:] for t in [self.observations,
                                                      self.actions,
                                                      self.rewards,
                                                      self.dones,
                                                      self.policy_logits,
                                                      self.values,
                                                      self.value_estimates])
        else:
            self.observations, self.actions, self.rewards, self.dones = [], [], [], []
            self.policy_logits, self.values, self.value_estimates = [], [], []

    def append(self, vec_obs, vec_a, vec_r, vec_d, vec_pl, vec_v, vec_v_est):
        self.observations.append(vec_obs)
        self.actions.append(vec_a)
        self.rewards.append(vec_r)
        self.dones.append(vec_d)
        self.policy_logits.append(vec_pl)
        self.values.append(vec_v)
        self.value_estimates.append(vec_v_est)

    def calculate_initial_priorities(self, rewards, values, value_estimates):
        """
        :param rewards: Stacked rewards for the buffer.
        :param value_targets: Stacked value targets for the buffer.
        :param value_estimates: Stacked initial value estimates for the buffer.
        :return: Value errors.
        """
        discounts = torch.ones_like(rewards)[:self.args.multistep]*self.args.discount
        discounts = discounts ** torch.arange(0, self.args.multistep)[:, None].float()

        valid_range = rewards[:-self.args.multistep].shape[0]
        discounted_rewards = torch.cat([rewards[i:i+self.args.multistep]*discounts for i in range(valid_range)], 0)
        value_targets = values[self.args.multistep:]*(self.args.discount ** self.args.multistep)

        value_targets = discounted_rewards + value_targets

        errors = torch.abs(value_estimates[:self.args.multistep] - value_targets) + 0.001

        return errors

    def stack(self):
        samples = [torch.stack(x) for x in [self.observations, self.actions,
                                            self.rewards, self.dones,
                                            self.policy_logits, self.values]]
        if self.args.input_priorities:
            priorities = self.calculate_initial_priorities(samples[2],
                                                           samples[5],
                                                           torch.stack(self.value_estimates))
            samples = [t[:-self.args.multistep] for t in samples]
            samples.append(priorities)

        return samples

# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree:
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = np.zeros((2 * size - 1,),
                                 dtype=np.float32)  # Initialise fixed size tree with all (priority) zeros
        self.data = np.array([None] * size)  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    def __len__(self):
        if self.full:
            return self.size
        else:
            return self.index

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

class ReplayMemory:
    def __init__(self, args, capacity, n=None, images=False, priority_exponent=None,
                 priority_weight=None, no_overshoot=False, no_segments=True):
        self.device = args.device
        self.capacity = capacity
        self.history = args.framestack
        self.discount = args.discount
        self.images = images
        if not n:
            self.n = args.multistep
        else:
            self.n = n

        self.priority_weight = priority_weight if priority_weight is not None else args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = priority_exponent if priority_exponent is not None else args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.no_overshoot = no_overshoot
        self.patience = 10  # How often to try sampling from a segment before choosing a different one.
        self.transitions = SegmentTree(
            capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.no_segments = no_segments

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, value, policy, terminal, timestep=None,
               init_priority=None):
        state = state[-1].to(device=torch.device('cpu'))
        if timestep is None:
            timestep = self.t
        if init_priority is None:
            init_priority = self.transitions.max
        self.transitions.append(Transition(timestep, state, action, reward, value, policy, 1 - terminal),
                                init_priority)  # Store new transition with maximum priority
        self.t = 0 if terminal is True else timestep + 1  # Start new episodes with t = 0

    # Returns a transition with blank states where appropriate
    def _get_transition(self, idx, n=None):
        if n is None:
            n = self.n
        transition = np.array([None] * (self.history + n))
        transition[self.history - 1] = self.transitions.get(idx)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            if transition[t + 1].timestep == 0:
                transition[t] = transition[t+1]  # If future frame has timestep 0
                transition[t].action = 0  # Pretend we did a no-op.
            else:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)

        for t in range(self.history, self.history + n):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
            else:
                # If we're terminal, just repeat the previous transition with
                # reward and value set to 0.
                transition[t] = transition[t - 1]  # If prev (next) frame is terminal
                transition[t].reward = 0.
                transition[t].value = 0.
        return transition

    def _sample_segment_with_intermediates(self, segment, i, n, batch_size):
        valid = False
        count = 0
        if self.no_segments:
            i = np.random.randint(0, batch_size)

        while not valid:
            sample = np.random.uniform(i * segment,
                                       (i + 1) * segment)  # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.transitions.find(
                sample)  # Retrieve sample from tree with un-normalised probability
            # Resample if transition straddled current index or probability 0
            if (self.transitions.index - idx) % self.capacity > n and (
                    idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
            else:
                count += 1
            if count > self.patience:
                i = np.random.randint(0, batch_size)

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx, n)
        # Discrete action to be used as index

        all_rewards = [trans.reward for trans in transition[self.history-1:-1]]
        all_actions = [trans.action for trans in transition]
        all_states = [trans.state for trans in transition]
        all_policies = [trans.policy for trans in transition[self.history-1:-1]]
        all_policies = torch.stack(all_policies, 0)
        all_values = [trans.value for trans in transition[self.history-1:-1]]

        return prob, idx, tree_idx, all_states, all_actions, all_rewards,\
               all_values, all_policies

    def sample(self, batch_size):
        n = self.n
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        batch = [self._sample_segment_with_intermediates(segment, i, n, batch_size) for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, all_states, all_actions, all_rewards, \
        all_values, all_policies = zip(*batch)
        probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32,
                               device=self.device)  # Normalise by max importance-sampling weight from batch

        all_actions = torch.tensor(all_actions, device=self.device).long()
        all_rewards = torch.tensor(all_rewards, device=self.device)
        all_values = torch.tensor(all_values, device=self.device)
        all_policies = torch.stack(all_policies, 0).to(self.device)
        all_states = [torch.stack(l, 0) for l in all_states]
        all_states = torch.stack(all_states).to(self.device)

        return tree_idxs, all_states, all_actions,\
               all_rewards, all_policies, all_values, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.history - 1)):
            if prev_timestep == 0:
                state_stack[t] = self.blank_trans  # If future frame has timestep 0
            else:
                state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
                prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(device=self.device)  # Agent will turn into batch
        self.current_idx += 1
        return state

    next = __next__  # Alias __next__ for Python 2 compatibility
