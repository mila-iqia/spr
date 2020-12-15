import torch

from rlpyt.utils.collections import namedarraytuple
from collections import namedtuple
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.logging import logger
from src.rlpyt_buffer import AsyncPrioritizedSequenceReplayFrameBufferExtended, \
    AsyncUniformSequenceReplayFrameBufferExtended
from src.models import from_categorical, to_categorical
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])
ModelSamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done", "value"])

OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
ModelOptInfo = namedtuple("OptInfo", ["loss", "gradNorm",
                                      "tdAbsErr",
                                      "modelRLLoss",
                                      "RewardLoss",
                                      "modelGradNorm",
                                      "SPRLoss",
                                      "ModelSPRLoss"])

EPS = 1e-6  # (NaN-guard)


class SPRCategoricalDQN(CategoricalDQN):
    """Distributional DQN with fixed probability bins for the Q-value of each
    action, a.k.a. categorical."""

    def __init__(self,
                 t0_spr_loss_weight=1.,
                 model_rl_weight=1.,
                 reward_loss_weight=1.,
                 model_spr_weight=1.,
                 time_offset=0,
                 distributional=1,
                 jumps=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.opt_info_fields = tuple(f for f in ModelOptInfo._fields)  # copy
        self.t0_spr_loss_weight = t0_spr_loss_weight
        self.model_spr_weight = model_spr_weight

        self.reward_loss_weight = reward_loss_weight
        self.model_rl_weight = model_rl_weight
        self.time_offset = time_offset
        self.jumps = jumps

        if not distributional:
            self.rl_loss = self.dqn_rl_loss
        else:
            self.rl_loss = self.dist_rl_loss

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        example_to_buffer = ModelSamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            value=examples["agent_info"].p,
        )
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            batch_T=self.jumps+1+self.time_offset,
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
            value=samples.agent.agent_info.p,
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
            t0_spr_loss, model_spr_loss = self.loss(samples_from_replay)
            spr_loss = self.t0_spr_loss_weight*t0_spr_loss + self.model_spr_weight*model_spr_loss
            total_loss = loss + self.model_rl_weight*model_rl_loss \
                              + self.reward_loss_weight*reward_loss
            total_loss = total_loss + spr_loss
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
            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(torch.tensor(grad_norm).item())  # grad_norm is a float sometimes, so wrap in tensor
            opt_info.modelRLLoss.append(model_rl_loss.item())
            opt_info.RewardLoss.append(reward_loss.item())
            opt_info.modelGradNorm.append(torch.tensor(model_grad_norm).item())
            opt_info.SPRLoss.append(spr_loss.item())
            opt_info.ModelSPRLoss.append(model_spr_loss.item())
            opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)
        self.update_itr_hyperparams(itr)
        return opt_info

    def dqn_rl_loss(self, qs, samples, index):
        """
        Computes the Q-learning loss, based on: 0.5 * (Q - target_Q) ^ 2.
        Implements regular DQN or Double-DQN for computing target_Q values
        using the agent's target network.  Computes the Huber loss using
        ``delta_clip``, or if ``None``, uses MSE.  When using prioritized
        replay, multiplies losses by importance sample weights.

        Input ``samples`` have leading batch dimension [B,..] (but not time).

        Calls the agent to compute forward pass on training inputs, and calls
        ``agent.target()`` to compute target values.

        Returns loss and TD-absolute-errors for use in prioritization.

        Warning:
            If not using mid_batch_reset, the sampler will only reset environments
            between iterations, so some samples in the replay buffer will be
            invalid.  This case is not supported here currently.
        """
        q = select_at_indexes(samples.all_action[index+1], qs).cpu()
        with torch.no_grad():
            target_qs = self.agent.target(samples.all_observation[index + self.n_step_return],
                                          samples.all_action[index + self.n_step_return],
                                          samples.all_reward[index + self.n_step_return])  # [B,A,P']
            if self.double_dqn:
                next_qs = self.agent(samples.all_observation[index + self.n_step_return],
                                     samples.all_action[index + self.n_step_return],
                                     samples.all_reward[index + self.n_step_return])  # [B,A,P']
                next_a = torch.argmax(next_qs, dim=-1)
                target_q = select_at_indexes(next_a, target_qs)
            else:
                target_q = torch.max(target_qs, dim=-1).values

            disc_target_q = (self.discount ** self.n_step_return) * target_q
            y = samples.return_[index] + (1 - samples.done_n[index].float()) * disc_target_q

        delta = y - q
        losses = 0.5 * delta ** 2
        abs_delta = abs(delta)
        if self.delta_clip > 0:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            losses = torch.where(abs_delta <= self.delta_clip, losses, b)
        td_abs_errors = abs_delta.detach()
        if self.delta_clip > 0:
            td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)
        return losses, td_abs_errors

    def dist_rl_loss(self, log_pred_ps, samples, index):
        delta_z = (self.V_max - self.V_min) / (self.agent.n_atoms - 1)
        z = torch.linspace(self.V_min, self.V_max, self.agent.n_atoms)
        # Make 2-D tensor of contracted z_domain for each data point,
        # with zeros where next value should not be added.
        next_z = z * (self.discount ** self.n_step_return)  # [P']
        next_z = torch.ger(1 - samples.done_n[index].float(), next_z)  # [B,P']
        ret = samples.return_[index].unsqueeze(1)  # [B,1]
        next_z = torch.clamp(ret + next_z, self.V_min, self.V_max)  # [B,P']

        z_bc = z.view(1, -1, 1)  # [1,P,1]
        next_z_bc = next_z.unsqueeze(1)  # [B,1,P']
        abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
        projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)  # Most 0.
        # projection_coeffs is a 3-D tensor: [B,P,P']
        # dim-0: independent data entries
        # dim-1: base_z atoms (remains after projection)
        # dim-2: next_z atoms (summed in projection)

        with torch.no_grad():
            target_ps = self.agent.target(samples.all_observation[index + self.n_step_return],
                                          samples.all_action[index + self.n_step_return],
                                          samples.all_reward[index + self.n_step_return])  # [B,A,P']
            if self.double_dqn:
                next_ps = self.agent(samples.all_observation[index + self.n_step_return],
                                     samples.all_action[index + self.n_step_return],
                                     samples.all_reward[index + self.n_step_return])  # [B,A,P']
                next_qs = torch.tensordot(next_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(next_qs, dim=-1)  # [B]
            else:
                target_qs = torch.tensordot(target_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(target_qs, dim=-1)  # [B]
            target_p_unproj = select_at_indexes(next_a, target_ps)  # [B,P']
            target_p_unproj = target_p_unproj.unsqueeze(1)  # [B,1,P']
            target_p = (target_p_unproj * projection_coeffs).sum(-1)  # [B,P]
        p = select_at_indexes(samples.all_action[index + 1].squeeze(-1),
                              log_pred_ps.cpu())  # [B,P]
        # p = torch.clamp(p, EPS, 1)  # NaN-guard.
        losses = -torch.sum(target_p * p, dim=1)  # Cross-entropy.

        target_p = torch.clamp(target_p, EPS, 1)
        KL_div = torch.sum(target_p *
            (torch.log(target_p) - p.detach()), dim=1)
        KL_div = torch.clamp(KL_div, EPS, 1 / EPS)  # Avoid <0 from NaN-guard.

        return losses, KL_div.detach()

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
        log_pred_ps, pred_rew, spr_loss\
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

        if self.model_rl_weight > 0:
            for i in range(1, self.jumps+1):
                    jump_rl_loss, model_KL = self.rl_loss(log_pred_ps[i],
                                                   samples,
                                                   i)
                    model_rl_loss = model_rl_loss + jump_rl_loss

        nonterminals = 1. - torch.sign(torch.cumsum(samples.done.to(self.agent.device), 0)).float()
        nonterminals = nonterminals[self.model.time_offset:
                                    self.jumps + self.model.time_offset+1]
        spr_loss = spr_loss*nonterminals
        if self.jumps > 0:
            model_spr_loss = spr_loss[1:].mean(0)
            spr_loss = spr_loss[0]
        else:
            spr_loss = spr_loss[0]
            model_spr_loss = torch.zeros_like(spr_loss)
        spr_loss = spr_loss.cpu()
        model_spr_loss = model_spr_loss.cpu()
        reward_loss = reward_loss.cpu()
        if self.prioritized_replay:
            weights = samples.is_weights
            spr_loss = spr_loss * weights
            model_spr_loss = model_spr_loss * weights
            reward_loss = reward_loss * weights

            # RL losses are no longer scaled in the c51 function
            rl_loss = rl_loss * weights
            model_rl_loss = model_rl_loss * weights

        return rl_loss.mean(), KL, \
               model_rl_loss.mean(),\
               reward_loss.mean(), \
               spr_loss.mean(), \
               model_spr_loss.mean(),
