from rlpyt.algos.dqn.dqn import DQN
import torch


class ValueLearning(DQN):
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
        log_pred_ps, pred_rew, mpr_loss \
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
            # RL losses are no longer scaled in the c51 function
            rl_loss = rl_loss * weights
            model_rl_loss = model_rl_loss * weights
        return rl_loss.mean(), KL, \
               model_rl_loss.mean(), \
               reward_loss.mean(), \
               mpr_loss.mean(), \
               model_mpr_loss.mean(),

    def rl_loss(self, samples):
        """
        Computes value loss for all timesteps in t = [0..jumps)
        """
        v = self.agent(*samples.agent_inputs)
        with torch.no_grad():
            target_v = self.agent.target(*samples.target_inputs)
        disc_target_v = (self.discount ** self.n_step_return) * target_v
        y = samples.return_ + (1 - samples.done_n.float()) * disc_target_v
        delta = y - v
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

