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
import traceback

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
        replay_kwargs['alpha'] = args.priority_exponent
        replay_kwargs['beta'] = args.priority_weight
        replay_kwargs["input_priorities"] = args.input_priorities
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
        while True:
            sampled_indices = False
            try:
                self._async_pull()  # Updates from writers.
                batch_T = self.batch_T
                T_idxs, B_idxs = self.sample_idxs(batch_B, batch_T)
                sampled_indices = True
                if self.rnn_state_interval > 1:
                    T_idxs = T_idxs * self.rnn_state_interval

                batch = self.extract_batch(T_idxs, B_idxs, self.batch_T)
                rewards = torch.from_numpy(extract_sequences(self.samples.reward, T_idxs-1, B_idxs, self.batch_T))
                dones = torch.from_numpy(extract_sequences(self.samples.done, T_idxs-1, B_idxs, self.batch_T + 1))
                policies = torch.from_numpy(extract_sequences(self.samples.policy_probs, T_idxs, B_idxs, self.batch_T))
                values = torch.from_numpy(extract_sequences(self.samples.value, T_idxs, B_idxs, self.batch_T))
                batch = list(batch)
                batch[2] = rewards
                batch[4] = dones
                batch = SamplesFromReplayExt(*batch, policy_probs=policies, values=values)
                return self.sanitize_batch(batch)
            except:
                print("FAILED TO LOAD BATCH")
                if sampled_indices:
                    print("B_idxs:", B_idxs, flush=True)
                    print("T_idxs:", T_idxs, flush=True)
                    print("Batch_T:", self.batch_T, flush=True)
                    print("Buffer T:", self.T, flush=True)

    def sanitize_batch(self, batch):
        has_dones, inds = torch.max(batch.done[1:], 0)
        for i, (has_done, ind) in enumerate(zip(has_dones, inds)):
            if not has_done:
                continue
            batch.all_observation[ind+1:, i] = batch.all_observation[ind, i]
            batch.all_action[ind+1:, i] = batch.all_action[ind, i]
            batch.all_action[ind+1:, i] = batch.all_action[ind, i]
            batch.policy_probs[ind+1:, i] = batch.policy_probs[ind, i]
            batch.all_reward[ind+1:, i] = 0
            batch.values[ind:, i] = 0
        return batch


class AsyncPrioritizedSequenceReplayFrameBufferExtended(AsyncPrioritizedSequenceReplayFrameBuffer):
    """
    Extends AsyncPrioritizedSequenceReplayFrameBuffer to return policy_logits and values too during sampling.
    """
    def sample_batch(self, batch_B):
        while True:
            sampled_indices = False
            try:
                self._async_pull()  # Updates from writers.
                (T_idxs, B_idxs), priorities = self.priority_tree.sample(
                    batch_B, unique=self.unique)
                sampled_indices = True
                if self.rnn_state_interval > 1:
                    T_idxs = T_idxs * self.rnn_state_interval

                batch = self.extract_batch(T_idxs, B_idxs, self.batch_T)
                is_weights = (1. / (priorities + 1e-5)) ** self.beta
                is_weights /= max(is_weights)  # Normalize.
                is_weights = torchify_buffer(is_weights).float()

                rewards = torch.from_numpy(extract_sequences(self.samples.reward, T_idxs-1, B_idxs, self.batch_T))
                dones = torch.from_numpy(extract_sequences(self.samples.done, T_idxs-1, B_idxs, self.batch_T + 1))
                policies = torch.from_numpy(extract_sequences(self.samples.policy_probs, T_idxs, B_idxs, self.batch_T))
                values = torch.from_numpy(extract_sequences(self.samples.value, T_idxs, B_idxs, self.batch_T))
                batch = list(batch)
                batch[2] = rewards
                batch[4] = dones
                batch = SamplesFromReplayPriExt(*batch, is_weights=is_weights, policy_probs=policies, values=values)
                return self.sanitize_batch(batch)
            except Exception as e:
                print("FAILED TO LOAD BATCH")
                traceback.print_exc()
                if sampled_indices:
                    print("B_idxs:", B_idxs, flush=True)
                    print("T_idxs:", T_idxs, flush=True)
                    print("Batch_T:", self.batch_T, flush=True)
                    print("Buffer T:", self.T, flush=True)

    def update_batch_priorities(self, priorities):
        with self.rw_lock.write_lock:
            priorities = numpify_buffer(priorities)
            self.default_priority = max(priorities)
            self.priority_tree.update_batch_priorities(priorities ** self.alpha)

    def sanitize_batch(self, batch):
        has_dones, inds = torch.max(batch.done[1:], 0)
        for i, (has_done, ind) in enumerate(zip(has_dones, inds)):
            if not has_done:
                continue
            batch.all_observation[ind+1:, i] = batch.all_observation[ind, i]
            batch.all_action[ind+1:, i] = batch.all_action[ind, i]
            batch.all_action[ind+1:, i] = batch.all_action[ind, i]
            batch.policy_probs[ind+1:, i] = batch.policy_probs[ind, i]
            batch.all_reward[ind+1:, i] = 0
            batch.values[ind:, i] = 0
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

    def calculate_initial_priorities(self, rewards, values, value_estimates, dones):
        """
        :param rewards: Stacked rewards for the buffer.
        :param value_targets: Stacked value targets for the buffer.
        :param value_estimates: Stacked initial value estimates for the buffer.
        :return: Value errors.
        """
        valid_range = rewards[:-self.args.multistep].shape[0]
        dones = 1 - dones.float()
        done_masks = torch.stack([torch.cumprod(dones[i:i+self.args.multistep+1], 0) for i in range(valid_range)])
        discounts = torch.ones_like(rewards)[:self.args.multistep+1]*self.args.discount
        discounts = discounts ** torch.arange(0, self.args.multistep+1)[:, None].float()
        discounts = discounts

        discounted_rewards = torch.stack([torch.sum(discounts[:-1]*done_masks[i, :-1]*rewards[i:i+self.args.multistep], 0) for i in range(valid_range)], 0)
        value_targets = values[self.args.multistep:]*discounts[-1]*done_masks[:, -1]

        value_targets = discounted_rewards + value_targets

        errors = torch.abs(value_estimates[self.args.multistep:] - value_targets) + 0.001

        return errors

    def stack(self):
        samples = [torch.stack(x) for x in [self.observations, self.actions,
                                            self.rewards, self.dones,
                                            self.policy_logits, self.values]]
        if self.args.input_priorities:
            priorities = self.calculate_initial_priorities(samples[2],
                                                           samples[5],
                                                           torch.stack(self.value_estimates),
                                                           samples[3])
            samples = [t[:-self.args.multistep] for t in samples]
            samples.append(priorities)

        return samples


