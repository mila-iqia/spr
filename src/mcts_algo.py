
import torch
import torch.nn.functional as F

from rlpyt.algos.dqn.dqn import DQN
from rlpyt.utils.collections import namedarraytuple
from collections import namedtuple
from src.rlpyt_buffer import AsyncPrioritizedSequenceReplayFrameBufferExtended, \
    AsyncUniformSequenceReplayFrameBufferExtended
from src.mcts_models import to_categorical
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])
ModelSamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done", "value", "policy"])
import time

OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
ModelOptInfo = namedtuple("OptInfo", ["loss", "gradNorm",
                                      "tdAbsErr",
                                      "modelRLLoss",
                                      "RewardLoss",
                                      "modelGradNorm",
                                      "PolicyLoss",
                                      "MPRLoss",
                                      "ModelMPRLoss"])

EPS = 1e-6  # (NaN-guard)


class ValueLearning(DQN):
    def __init__(self,
                 t0_mpr_loss_weight=1.,
                 model_rl_weight=1.,
                 reward_loss_weight=1.,
                 model_mpr_weight=1.,
                 policy_loss_weight=1.,
                 jumps=0,
                 discount=0.99,
                 n_step_return=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.opt_info_fields = tuple(f for f in ModelOptInfo._fields)  # copy
        self.t0_mpr_loss_weight = t0_mpr_loss_weight
        self.model_mpr_weight = model_mpr_weight
        self.policy_loss_weight = policy_loss_weight

        self.reward_loss_weight = reward_loss_weight
        self.model_rl_weight = model_rl_weight
        self.jumps = jumps
        self.discount = discount
        self.n_step_return = n_step_return

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        example_to_buffer = ModelSamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            value=examples["agent_info"].value,
            policy=examples["agent_info"].policy,
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
            value=samples.agent.agent_info.value,
            policy=samples.agent.agent_info.policy
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
            loss, td_abs_errors, model_rl_loss, reward_loss,\
            t0_mpr_loss, model_mpr_loss, policy_loss = self.loss(samples_from_replay)
            mpr_loss = self.t0_mpr_loss_weight*t0_mpr_loss + self.model_mpr_weight*model_mpr_loss
            total_loss = loss + self.model_rl_weight*model_rl_loss \
                              + self.reward_loss_weight*reward_loss \
                              + self.policy_loss_weight*policy_loss
            total_loss = total_loss + mpr_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.stem_parameters(), self.clip_grad_norm)
            if len(list(self.model.dynamics_model.parameters())) > 0:
                model_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.dynamics_model.parameters(), self.clip_grad_norm)
            else:
                model_grad_norm = 0
            self.optimizer.step()
            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(td_abs_errors)
            # print(self.update_counter, mpr_loss, model_grad_norm)
            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(torch.tensor(grad_norm).item())  # grad_norm is a float sometimes, so wrap in tensor
            opt_info.modelRLLoss.append(model_rl_loss.item())
            opt_info.RewardLoss.append(reward_loss.item())
            opt_info.PolicyLoss.append(policy_loss.item())
            opt_info.modelGradNorm.append(torch.tensor(model_grad_norm).item())
            opt_info.MPRLoss.append(t0_mpr_loss.item())
            opt_info.ModelMPRLoss.append(model_mpr_loss.item())
            opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)
        self.update_itr_hyperparams(itr)
        return opt_info

    def loss(self, samples):
        """
        Computes the Distributional Q-learning loss, based on projecting the
        discounted rewards + target Q-distribution into the current Q-domain,
        with cross-entropy loss.
        Returns loss and KL-divergence-errors for use in prioritization.
        """
        if self.model.noisy:
            self.model.head.reset_noise()
        # start = time.time()
        log_pred_ps, pred_rew, mpr_loss, pred_policy_logits \
            = self.agent(samples.all_observation.to(self.agent.device),
                         samples.all_action.to(self.agent.device),
                         samples.all_reward.to(self.agent.device),
                         train=True)  # [B,A,P]
        rl_loss, KL = self.rl_loss(log_pred_ps[0], samples, 0)
        if len(pred_rew) > 0:
            pred_rew = torch.stack(pred_rew, 0)
            with torch.no_grad():
                reward_target = to_categorical(samples.all_reward[:self.jumps+1].flatten().to(self.agent.device), limit=1).view(*pred_rew.shape)
            reward_loss = -torch.sum(reward_target * pred_rew, 2).mean(0).cpu()
        else:
            reward_loss = torch.zeros(samples.all_observation.shape[1],)
        model_rl_loss = torch.zeros_like(reward_loss)

        pred_policy_logits = torch.stack(pred_policy_logits, 0).cpu()  # T, B, Actions
        policy_loss = -torch.sum(samples.policy*pred_policy_logits, 2).mean(0)

        if self.model_rl_weight > 0:
            for i in range(1, self.jumps+1):
                jump_rl_loss, _ = self.rl_loss(log_pred_ps[i],
                                               samples,
                                               i)
                model_rl_loss = model_rl_loss + jump_rl_loss

        nonterminals = 1. - torch.sign(torch.cumsum(samples.done.to(self.agent.device), 0)).float()
        nonterminals = nonterminals[self.model.time_offset:
                                    self.jumps + self.model.time_offset+1]
        mpr_loss = mpr_loss*nonterminals
        if self.jumps > 0:
            model_mpr_loss = mpr_loss[1:].mean(0)
            mpr_loss = mpr_loss[0]
        else:
            mpr_loss = mpr_loss[0]
            model_mpr_loss = torch.zeros_like(mpr_loss)
        mpr_loss = mpr_loss.cpu()
        model_mpr_loss = model_mpr_loss.cpu()
        reward_loss = reward_loss.cpu()

        if self.prioritized_replay:
            weights = samples.is_weights
            mpr_loss = mpr_loss * weights
            model_mpr_loss = model_mpr_loss * weights
            reward_loss = reward_loss * weights
            policy_loss = policy_loss * weights
            # RL losses are no longer scaled in the c51 function
            rl_loss = rl_loss * weights
            model_rl_loss = model_rl_loss * weights
        return rl_loss.mean(), KL, \
               model_rl_loss.mean(), \
               reward_loss.mean(), \
               mpr_loss.mean(), \
               model_mpr_loss.mean(), \
               policy_loss.mean()

    def rl_loss(self, pred_v, samples, index):
        """
        Computes value loss for all timesteps in t = [0..jumps)
        """
        with torch.no_grad():
            target_v = self.agent.target(samples.all_observation[index + self.n_step],
                                         samples.all_action[index + self.n_step],
                                         samples.all_reward[index + self.n_step])
        disc_target_v = (self.discount ** self.n_step_return) * target_v
        y = samples.return_[index] + (1 - samples.done_n[index].float()) * disc_target_v
        delta = y - pred_v
        value_losses = 0.5 * delta ** 2
        abs_delta = abs(delta)
        if self.delta_clip is not None:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            value_losses = torch.where(abs_delta <= self.delta_clip, value_losses, b)
        if self.prioritized_replay:
            value_losses *= samples.is_weights
        td_abs_errors = abs_delta.detach()
        if self.delta_clip is not None:
            td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)
        value_loss = torch.mean(value_losses)
        return value_loss, td_abs_errors

