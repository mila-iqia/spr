import torch
import torch.nn.functional as F
import numpy as np

from rlpyt.utils.collections import namedarraytuple
from collections import namedtuple
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.logging import logger
from src.buffer import AsyncPrioritizedSequenceReplayFrameBufferExtended, \
    AsyncUniformSequenceReplayFrameBufferExtended
from rlpyt.algos.utils import discount_return_n_step
from src.utils import minimal_c51_loss, average_targets, \
    select_at_indexes, timeit, safe_log, to_categorical, \
    from_categorical, c51_backup, scalar_backup, minimal_scalar_loss, \
    minimal_quantile_loss
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])
ModelSamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done", "value"])

OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
ModelOptInfo = namedtuple("OptInfo", ["loss", "gradNorm",
                                      "tdAbsErr",
                                      "ModelRLLoss",
                                      "RolloutRLLoss",
                                      "DoneLoss",
                                      "Collapse",
                                      "SampledLambda",
                                      "OptimLambda",
                                      "LambdaLoss",
                                      "RewardLoss",
                                      "modelGradNorm",
                                      "SPRLoss",
                                      "CalibrationLoss",
                                      "SearchTemperature",
                                      "ModelSPRLoss"])

EPS = 1e-6  # (NaN-guard)


class LambdaBuffer:
    def __init__(self,
                 buffer_size=1000,):
        self.size = buffer_size

        self.full = False
        self.index = 0
        self.initialized = False

    def create_buffers(self,
                       shapes):
        """
        For simplicity, we create the buffers lazily once we actually
        get given a set of inputs, so that we don't have to determine
        shapes in advance.
        :param shapes
        :return:
        """
        self.buffers = [torch.zeros((self.size, *shape)) for shape in shapes]

    def append(self, tensors):
        if not self.initialized:
            self.device = tensors[0].device
            self.create_buffers([t.shape for t in tensors])
            self.initialized = True
        for buffer, tensor in zip(self.buffers, tensors):
            buffer[self.index] = tensor.cpu().detach()

        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True

    def extend(self, tensors):
        if not self.initialized:
            self.device = tensors[0].device
            self.create_buffers([t.shape[1:] for t in tensors])
            self.initialized = True

        update_size = tensors[0].shape[0]

        if self.index + update_size > self.size:
            self.full = True
            gap = self.index + update_size - self.size
            self.extend([t[:-gap] for t in tensors])
            self.extend([t[-gap:] for t in tensors])
        else:
            for buffer, tensor in zip(self.buffers, tensors):
                buffer[self.index: self.index + update_size] = tensor.cpu().detach()
            self.index = (self.index + update_size) % self.size

    def current_samples(self):
        return self.size if self.full else self.index

    def sample(self, batch_size=32,):
        indices = np.random.randint(low=0,
                                    high=self.current_samples(),
                                    size=(batch_size,))

        return [buffer[indices].to(self.device) for buffer in self.buffers]


class SPRCategoricalDQN(CategoricalDQN):
    """Distributional DQN with fixed probability bins for the Q-value of each
    action, a.k.a. categorical."""

    def __init__(self,
                 t0_spr_loss_weight=1.,
                 model_rl_weight=1.,
                 counterfactual_runs=1,
                 reward_loss_weight=1.,
                 model_spr_weight=1.,
                 distributional=1,
                 rollout_rl_weight=1.,
                 done_loss_weight=1.,
                 online_buffer_size=1000,
                 lambda_rollout_depth=5,
                 rl_weight=1.,
                 transfer_freq=16,
                 jumps=0,
                 debug_step=-1,
                 counterfactual_from_obs=1,
                 counterfactual_with_target=1,
                 reset_target_noise=True,
                 rollout_rl_type="offset",
                 use_dones=True,
                 search_start_offset=0,
                 head_only_search=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.opt_info_fields = tuple(f for f in ModelOptInfo._fields)  # copy
        self.t0_spr_loss_weight = t0_spr_loss_weight
        self.model_spr_weight = model_spr_weight

        self.reward_loss_weight = reward_loss_weight
        self.model_rl_weight = model_rl_weight
        self.counterfactual_runs = counterfactual_runs
        self.rollout_rl_type = rollout_rl_type
        self.jumps = jumps
        self.done_loss_weight = done_loss_weight*float(use_dones)
        self.use_dones = use_dones
        self.online_buffer_size = online_buffer_size
        self.lambda_rollout_depth = lambda_rollout_depth
        self.rollout_rl_weight = rollout_rl_weight
        self.rl_weight = rl_weight
        self.reset_target_noise = reset_target_noise
        self.debug_step = debug_step
        self.last_transfer = 0
        self.transfer_freq = transfer_freq
        self.counterfactual_from_obs = counterfactual_from_obs
        self.counterfactual_with_target = counterfactual_with_target
        self.head_only_search = head_only_search
        self.search_start_offset = search_start_offset

        self.distributional = distributional != "quantile" and distributional
        self.quantile = distributional == "quantile"

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
            batch_T=self.jumps + 1,
            discount=self.discount,
            n_step_return=self.n_step_return,
            rnn_state_interval=0,
        )

        self.batch_B = batch_spec.B

        if self.prioritized_replay:
            replay_kwargs['alpha'] = self.pri_alpha
            replay_kwargs['beta'] = self.pri_beta_init
            # replay_kwargs["input_priorities"] = self.input_priorities
            buffer = AsyncPrioritizedSequenceReplayFrameBufferExtended(**replay_kwargs)
        else:
            buffer = AsyncUniformSequenceReplayFrameBufferExtended(**replay_kwargs)

        self.online_buffer = LambdaBuffer(self.online_buffer_size,)
        self.replay_buffer = buffer
        return buffer

    @torch.no_grad()
    def transfer_samples(self, t_range=1):
        depth = self.lambda_rollout_depth
        if depth == 0:
            return
        t_idxs = torch.tensor([self.replay_buffer.t - depth - i for i in range(t_range)]*self.batch_B, dtype=torch.long)
        if torch.min(t_idxs) < 0:
            return
        b_idxs = torch.cat([torch.arange(self.batch_B)] * t_range, 0)
        new_samples = self.replay_buffer.extract_batch(t_idxs, b_idxs, depth+1)

        mode = self.agent._mode
        self.agent.quiet_eval()

        input = new_samples.all_observation.to(self.agent.device)[:1]
        actions = new_samples.all_action[1:].to(self.agent.device)
        if self.quantile:
            self.model.taus = torch.linspace(0, 1, 34, device=input.device)[None, 1:-1].expand(input.shape[1], -1)
        latents, ps, _, _, _, _, _ = self.model.model_rollout(input,
                                                              depth,
                                                              override_actions=actions,
                                                              encode=True,
                                                              augment=False,
                                                              )

        target_inputs = (new_samples.all_observation[depth].to(self.agent.device),
                         new_samples.all_action[depth].to(self.agent.device),
                         new_samples.all_reward[depth].to(self.agent.device),)
        target_qs = self.model(*target_inputs,
                               tau_override=self.model.taus,
                               eval=False,
                               train=False,
                               force_no_rollout=True,
                              )

        return_, done_ = discount_return_n_step(new_samples.all_reward[1:depth+1],
                                                new_samples.done,
                                                depth,
                                                self.discount)

        return_ = return_.to(target_qs)[0]
        done_ = done_.to(target_qs)[0]
        
        target_qs = self.backup(depth, return_, 1-done_, target_qs, select_action=True)

        buffer_input = [input[0], ps, target_qs.squeeze()]
        if self.quantile:
            buffer_input.append(self.model.taus)

        self.agent.set_mode(mode)
        self.online_buffer.extend(buffer_input)

    def model_backup_loss(self, observation, latent, rollouts=1,):
        if self.head_only_search:
            latent = latent.detach()
        all_pred_ps = self.model.head_forward(latent, logits=False)

        repeated_pred_ps = all_pred_ps.unsqueeze(1).expand(-1, rollouts, *([-1]*len(all_pred_ps.shape[1:]))).flatten(0, 1)
        with torch.no_grad():
            if self.counterfactual_from_obs:
                repeated_inputs = observation.unsqueeze(1).expand(-1, rollouts, *([-1]*len(observation.shape[1:]))).flatten(0, 1)
                repeated_inputs = repeated_inputs.unsqueeze(0)
            else:
                repeated_inputs = latent.unsqueeze(1).expand(-1, rollouts, *([-1]*len(latent.shape[1:]))).flatten(0, 1)
            initial_actions, _ = self.model.action_selection(all_pred_ps,
                                                             temperature=0.5,
                                                             adjust_temp=False,
                                                             num_to_select=rollouts)
            if self.counterfactual_with_target:
                model = self.agent.target_model
                if self.quantile:
                    self.agent.target_model.taus = self.model.taus
            else:
                model = self.agent.model
            model.eval()
            _, target_ps, _, actions, _, _, lambdas = \
                model.model_rollout(repeated_inputs,
                                    encode=self.counterfactual_from_obs,
                                    augment=self.counterfactual_from_obs,
                                    override_actions=[initial_actions],
                                    depth=self.model.target_depth,
                                    backup=True,
                                    adaptive_depth=True,
                                    runs=1,
                                    )
            model.train()
            target_ps[:, 0] = select_at_indexes(initial_actions, repeated_pred_ps)
            targets = average_targets(target_ps, lambdas)
        pred_ps = select_at_indexes(initial_actions, repeated_pred_ps)
        loss, _ = self.minimal_rl_loss(pred_ps, targets)
        loss = loss.view(all_pred_ps.shape[0], -1, *loss.shape[1:]).mean(1)
        return loss

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
            self.last_transfer += 1
        opt_info = ModelOptInfo(*([] for _ in range(len(ModelOptInfo._fields))))
        if itr == self.debug_step:
            self.model.debug = True

        if self.last_transfer >= self.transfer_freq:
            self.transfer_samples(self.transfer_freq)
            self.last_transfer = 0

        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):

            if self.online_buffer.current_samples() > 50 and self.lambda_rollout_depth > 0:
                lambda_loss, optim_lambdas = self.model.lambda_predictor.optimize(self.online_buffer)
            else:
                lambda_loss = 0
                optim_lambdas = np.zeros(32)

            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)

            rl_loss, \
            td_abs_errors, \
            model_rl_loss, \
            rollout_rl_loss, \
            reward_loss, \
            done_loss, \
            t0_spr_loss, \
            model_spr_loss, \
            diversity, \
            calibration_loss, \
            lambdas = self.loss(samples_from_replay)

            spr_loss = self.t0_spr_loss_weight*t0_spr_loss +\
                       self.model_spr_weight*model_spr_loss
            total_loss = self.rl_weight*rl_loss \
                       + self.model_rl_weight*model_rl_loss \
                       + self.rollout_rl_weight*rollout_rl_loss \
                       + self.reward_loss_weight*reward_loss \
                       + self.done_loss_weight*done_loss
            total_loss = total_loss + spr_loss + calibration_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.stem_parameters(), self.clip_grad_norm)
            if len(list(self.model.dynamics_model.parameters())) > 0:
                model_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.model_parameters(), self.clip_grad_norm)
            else:
                model_grad_norm = 0
            self.optimizer.step()
            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(td_abs_errors)

            # print(self.update_counter, spr_loss, model_grad_norm)
            opt_info.loss.append(rl_loss.item())
            opt_info.gradNorm.append(torch.tensor(grad_norm).item())  # grad_norm is a float sometimes, so wrap in tensor
            opt_info.ModelRLLoss.append(model_rl_loss.item())
            opt_info.RolloutRLLoss.append(rollout_rl_loss.item())
            opt_info.RewardLoss.append(reward_loss.item())
            opt_info.LambdaLoss.append(lambda_loss)
            opt_info.DoneLoss.append(done_loss.item())
            opt_info.Collapse.append(diversity.cpu().item())
            opt_info.CalibrationLoss.append(calibration_loss.cpu().item())
            opt_info.SearchTemperature.append(self.model.selection_temp.cpu().item())
            opt_info.SampledLambda.extend(lambdas[::8].cpu().numpy())
            opt_info.OptimLambda.extend(optim_lambdas[::8].cpu().numpy())
            opt_info.modelGradNorm.append(torch.tensor(model_grad_norm).item())
            opt_info.SPRLoss.append(t0_spr_loss.item())
            opt_info.ModelSPRLoss.append(model_spr_loss.item())
            opt_info.tdAbsErr.extend(td_abs_errors[::8].cpu().numpy())  # Downsample.
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)
        self.update_itr_hyperparams(itr)
        return opt_info

    def rl_loss(self, pred_ps, samples, indices
                ):
        with torch.no_grad():
            next_obses = []
            next_actions = []
            next_rewards = []
            returns = []
            done_ns = []
            n_step = self.n_step_return
            for index in indices:
                next_obses.append(samples.all_observation[index+n_step])
                next_actions.append(samples.all_action[index+n_step])
                next_rewards.append(samples.all_reward[index+n_step])
                returns.append(samples.return_[index])
                done_ns.append(samples.done_n[index])
            target_obs = torch.stack(next_obses, 1).to(pred_ps.device).flatten(0, 1)
            target_action = torch.stack(next_actions, 1).to(pred_ps.device).flatten(0, 1)
            target_rewards = torch.stack(next_rewards, 1).to(pred_ps).flatten(0, 1)
            done_n = torch.stack(done_ns, 1).to(pred_ps).flatten(0, 1)
            return_ = torch.stack(returns, 1).to(pred_ps).flatten(0, 1)

            if self.double_dqn:
                next_ps = self.agent(target_obs,
                                     target_action,
                                     target_rewards,
                                     force_no_rollout=True)  # [B,A,P']
            else:
                next_ps = None

            target_ps = self.agent.target(target_obs,
                                          target_action,
                                          target_rewards,
                                          seed_actions=None)  # [B,A,P']

            target_p = self.backup(n_step, return_, 1-done_n,
                                   target_ps, select_action=True,
                                   selection_ps=next_ps
                                   )

        if len(pred_ps.shape) == 3:  # Stack
            target_p = target_p.unsqueeze(1)

        loss, td_error = self.minimal_rl_loss(pred_ps, target_p)

        return loss, td_error.detach()

    def backup(self, n_step, returns, nonterminal, target_ps, select_action=False,
               selection_ps=None):
        if self.distributional:
            return c51_backup(n_step, returns, nonterminal,
                              target_ps, select_action=select_action,
                              V_max=self.V_max,
                              V_min=self.V_min, discount=self.discount,
                              selection_values=selection_ps)
        else:
            return scalar_backup(n_step, returns, nonterminal, target_ps,
                                 discount=self.discount,
                                 select_action=select_action,
                                 selection_values=selection_ps)

    def minimal_rl_loss(self, preds, targets):
        if self.distributional:
            loss, delta = minimal_c51_loss(preds, targets)
        elif self.quantile:
            taus = self.model.taus
            loss, delta = minimal_quantile_loss(preds, targets, taus)
        else:
            loss, delta = minimal_scalar_loss(preds, targets, self.delta_clip)

        return loss, delta

    def loss(self, samples):
        """
        Computes the Distributional Q-learning loss, based on projecting the
        discounted rewards + target Q-distribution into the current Q-domain,
        with cross-entropy loss.

        Returns loss and KL-divergence-errors for use in prioritization.
        """
        if self.model.noisy:
            self.model.head.reset_noise()
            if self.reset_target_noise:
                self.agent.target_model.head.reset_noise()
        # start = time.time()
        observation = samples.all_observation.to(self.agent.device)
        pred_ps, pred_rew, pred_dones, spr_loss, diversity, lambdas,\
             pred_latents, calibration_loss \
            = self.agent(observation,
                         samples.all_action.to(self.agent.device),
                         samples.all_reward.to(self.agent.device),
                         train=True)  # [B,A,P]

        if self.rollout_rl_type == "offset":
            rl_loss, KL = self.rl_loss(pred_ps.flatten(0, 1), samples, range(0, self.jumps+1))
            rl_loss = rl_loss.view(pred_ps.shape[0], pred_ps.shape[1])
            KL = KL.view(pred_ps.shape[0], pred_ps.shape[1])
            KL = KL[:, 0]  # Prioritize based on initial loss
        elif self.rollout_rl_type == "backup":
            rl_loss, KL = self.rl_loss(pred_ps, samples, [0])
            KL = KL[:, 0]

        rl_loss, rollout_rl_loss = (rl_loss[:, 0], rl_loss[:, 1:].mean(1))

        with torch.no_grad():
            reward_target = to_categorical(samples.all_reward[:self.jumps+1].flatten().to(self.agent.device), limit=1).view(*pred_rew.shape)
        reward_loss = -torch.sum(reward_target * pred_rew, 2).mean(0)

        done_target = samples.done[:self.jumps+1].float().to(pred_dones.device)
        done_loss = F.binary_cross_entropy(pred_dones, done_target, reduction="none")
        done_loss = done_loss.mean(0)

        if self.model_rl_weight > 0:
            model_rl_loss = self.model_backup_loss(observation[self.search_start_offset],
                                                   pred_latents[:, self.search_start_offset],
                                                   rollouts=self.counterfactual_runs)
        else:
            model_rl_loss = torch.zeros_like(rl_loss)

        nonterminals = 1. - torch.sign(torch.cumsum(samples.done.to(self.agent.device), 0)).float()
        nonterminals = nonterminals[0: self.jumps + 1]
        spr_loss = spr_loss*nonterminals
        if self.jumps > 0:
            model_spr_loss = spr_loss[1:].mean(0)
            spr_loss = spr_loss[0]
        else:
            spr_loss = spr_loss[0]
            model_spr_loss = torch.zeros_like(spr_loss)
        reward_loss = reward_loss
        if self.prioritized_replay:
            weights = samples.is_weights.to(reward_loss.device)
            spr_loss = spr_loss * weights
            model_spr_loss = model_spr_loss * weights
            reward_loss = reward_loss * weights
            done_loss = done_loss * weights
            calibration_loss = calibration_loss*weights

            # RL losses are no longer scaled in the c51 function
            rl_loss = rl_loss * weights
            model_rl_loss = model_rl_loss * weights
            rollout_rl_loss = rollout_rl_loss * weights

        depth = torch.arange(lambdas.shape[-1], dtype=lambdas.dtype, device=lambdas.device)
        depth = (lambdas*(depth.unsqueeze(0))).sum(-1)

        return rl_loss.mean(), KL, \
               model_rl_loss.mean(),\
               rollout_rl_loss.mean(),\
               reward_loss.mean(), \
               done_loss.mean(), \
               spr_loss.mean(), \
               model_spr_loss.mean(), \
               diversity.mean(),\
               calibration_loss.mean(),\
               depth

