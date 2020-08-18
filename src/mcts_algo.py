from rlpyt.algos.dqn.dqn import DQN
import torch


class ValueLearning(DQN):
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
        if not self.mid_batch_reset:
            # FIXME: I think this is wrong, because the first "done" sample
            # is valid, but here there is no [T] dim, so there's no way to
            # know if a "done" sample is the first "done" in the sequence.
            raise NotImplementedError
            # valid = valid_from_done(samples.done)
            # loss = valid_mean(losses, valid)
            # td_abs_errors *= valid
        else:
            value_loss = torch.mean(value_losses)
        return value_loss

