import torch
import torch.nn.functional as F

from rlpyt.utils.collections import namedarraytuple
from collections import namedtuple
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done
from src.rlpyt_buffer import AsyncPrioritizedSequenceReplayFrameBufferExtended, \
    AsyncUniformSequenceReplayFrameBufferExtended
from src.model_trainer import from_categorical, to_categorical
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])

OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
ModelOptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr", "modelLoss", "modelGradNorm"])


EPS = 1e-6  # (NaN-guard)


class PizeroCategoricalDQN(CategoricalDQN):
    """Distributional DQN with fixed probability bins for the Q-value of each
    action, a.k.a. categorical."""

    def __init__(self, jumps=0, detach_model=True, **kwargs):
        """Standard __init__() plus Q-value limits; the agent configures
        the number of atoms (bins)."""
        self.jumps = jumps
        self.detach_model = detach_model
        super().__init__(**kwargs)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
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

    def loss(self, samples):
        """
        Computes the Distributional Q-learning loss, based on projecting the
        discounted rewards + target Q-distribution into the current Q-domain,
        with cross-entropy loss.

        Returns loss and KL-divergence-errors for use in prioritization.
        """
        delta_z = (self.V_max - self.V_min) / (self.agent.n_atoms - 1)
        z = torch.linspace(self.V_min, self.V_max, self.agent.n_atoms)
        # Make 2-D tensor of contracted z_domain for each data point,
        # with zeros where next value should not be added.
        next_z = z * (self.discount ** self.n_step_return)  # [P']
        next_z = torch.ger(1 - samples.done_n[0].float(), next_z)  # [B,P']
        ret = samples.return_[0].unsqueeze(1)  # [B,1]
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
            target_ps = self.agent.target(samples.all_observation[self.n_step_return],
                                          samples.all_action[self.n_step_return],
                                          samples.all_reward[self.n_step_return])  # [B,A,P']
            if self.double_dqn:
                next_ps = self.agent(samples.all_observation[self.n_step_return],
                                     samples.all_action[self.n_step_return],
                                     samples.all_reward[self.n_step_return])  # [B,A,P']
                next_qs = torch.tensordot(next_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(next_qs, dim=-1)  # [B]
            else:
                target_qs = torch.tensordot(target_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(target_qs, dim=-1)  # [B]
            target_p_unproj = select_at_indexes(next_a, target_ps)  # [B,P']
            target_p_unproj = target_p_unproj.unsqueeze(1)  # [B,1,P']
            target_p = (target_p_unproj * projection_coeffs).sum(-1)  # [B,P]
        ps = self.agent(samples.all_observation[0],
                        samples.all_action[0],
                        samples.all_reward[0])  # [B,A,P]
        p = select_at_indexes(samples.all_action[1], ps)  # [B,P]
        p = torch.clamp(p, EPS, 1)  # NaN-guard.
        losses = -torch.sum(target_p * torch.log(p), dim=1)  # Cross-entropy.

        if self.prioritized_replay:
            losses *= samples.is_weights

        target_p = torch.clamp(target_p, EPS, 1)
        KL_div = torch.sum(target_p *
            (torch.log(target_p) - torch.log(p.detach())), dim=1)
        KL_div = torch.clamp(KL_div, EPS, 1 / EPS)  # Avoid <0 from NaN-guard.

        if not self.mid_batch_reset:
            valid = valid_from_done(samples.done[0])
            loss = valid_mean(losses, valid)
            KL_div *= valid
        else:
            loss = torch.mean(losses)

        return loss, KL_div


class PizeroModelCategoricalDQN(PizeroCategoricalDQN):
    """Distributional DQN with fixed probability bins for the Q-value of each
    action, a.k.a. categorical."""

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        try:
            self.optimizer = self.OptimCls(self.agent.model.stem_parameters(),
                lr=self.learning_rate, **self.optim_kwargs)
            self.model_optimizer = self.OptimCls(self.agent.model.dynamics_model.parameters(),
                lr=self.learning_rate, **self.optim_kwargs)
            self.model = self.agent.model
        except:
            # We're probably dealing with DDP
            self.optimizer = self.OptimCls(self.agent.model.module.stem_parameters(),
                lr=self.learning_rate, **self.optim_kwargs)
            self.model_optimizer = self.OptimCls(self.agent.model.module.dynamics_model.parameters(),
                lr=self.learning_rate, **self.optim_kwargs)
            self.model = self.agent.model.module
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        if self.prioritized_replay:
            self.pri_beta_itr = max(1, self.pri_beta_steps // self.sampler_bs)

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Extracts the needed fields from input samples and stores them in the
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).  If using prioritized
        replay, updates the priorities for sampled training batches.
        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = ModelOptInfo(*([] for _ in range(len(ModelOptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            loss, td_abs_errors, model_loss = self.loss(samples_from_replay)
            total_loss = loss + model_loss
            self.optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.stem_parameters(), self.clip_grad_norm)
            model_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.dynamics_model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.model_optimizer.step()
            # else:
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     grad_norm = torch.nn.utils.clip_grad_norm_(
            #         self.model.stem_parameters(), self.clip_grad_norm)
            #     self.optimizer.step()
            #     self.model_optimizer.zero_grad()
            #     model_loss.backward()
            #     model_grad_norm = torch.nn.utils.clip_grad_norm_(
            #         self.model.dynamics_model.parameters(), self.clip_grad_norm)
            #     self.model_optimizer.step()

            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(td_abs_errors)
            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(grad_norm)
            opt_info.modelLoss.append(model_loss.item())
            opt_info.modelGradNorm.append(model_grad_norm)
            opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)
        self.update_itr_hyperparams(itr)
        return opt_info

    def rl_loss(self, pred_ps, samples, index):
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
                              pred_ps.cpu())  # [B,P]
        # p = torch.clamp(p, EPS, 1)  # NaN-guard.
        losses = -torch.sum(target_p * p, dim=1)  # Cross-entropy.

        if self.prioritized_replay:
            losses = losses*samples.is_weights

        target_p = torch.clamp(target_p, EPS, 1)
        KL_div = torch.sum(target_p *
            (torch.log(target_p) - p.detach()), dim=1)
        KL_div = torch.clamp(KL_div, EPS, 1 / EPS)  # Avoid <0 from NaN-guard.

        if not self.mid_batch_reset:
            valid = valid_from_done(samples.done[index])
            loss = valid_mean(losses, valid)
            KL_div *= valid
        else:
            loss = torch.mean(losses)

        return loss, KL_div

    def loss(self, samples):
        """
        Computes the Distributional Q-learning loss, based on projecting the
        discounted rewards + target Q-distribution into the current Q-domain,
        with cross-entropy loss.

        Returns loss and KL-divergence-errors for use in prioritization.
        """
        pred_ps, pred_rew = self.agent(samples.all_observation.to(self.agent.device),
                                       samples.all_action.to(self.agent.device),
                                       samples.all_reward.to(self.agent.device),
                                       jumps=True)  # [B,A,P]
        rl_loss, KL = self.rl_loss(pred_ps[0], samples, 0)
        pred_rew = torch.stack(pred_rew, 0)
        with torch.no_grad():
            reward_target = to_categorical(samples.all_reward[:self.jumps+1].flatten().to(self.agent.device), limit=1).view(*pred_rew.shape)
        model_loss = -torch.sum(reward_target * pred_rew, (0, 2))

        for i in range(1, self.jumps+1):
            jump_rl_loss, _ = self.rl_loss(pred_ps[i],
                                           samples,
                                           i)
            model_loss = model_loss + jump_rl_loss.to(self.agent.device)

        if self.prioritized_replay:
            model_loss = model_loss * samples.is_weights.to(self.agent.device)

        return rl_loss, KL, model_loss.mean().cpu()
