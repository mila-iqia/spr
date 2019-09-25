import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.data import RandomSampler, BatchSampler
import torch.nn.functional as F

from src.episodes import get_framestacked_transition


class ForwardModel:
    def __init__(self, args, encoder, num_actions):
        self.args = args
        self.device = args.device
        self.encoder = encoder
        self.num_actions = num_actions
        hidden_size = args.forward_hidden_size
        self.hidden = nn.Linear(args.feature_size * 4 * num_actions, hidden_size).to(self.device)
        self.sd_predictor = nn.Linear(hidden_size, args.feature_size).to(self.device)
        self.reward_predictor = nn.Linear(hidden_size, 3).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.hidden.parameters()) +
                                          list(self.sd_predictor.parameters()) +
                                          list(self.reward_predictor.parameters()))

    def generate_reward_class_weights(self, transitions):
        counts = [0, 0, 0]  # counts for reward=-1,0,1
        for trans in transitions:
            counts[trans.reward + 1] += 1

        weights = [0., 0., 0.]
        for i in range(3):
            if counts[i] != 0:
                weights[i] = sum(counts) / counts[i]
        return torch.tensor(weights, device=self.device)

    def generate_batch(self, transitions):
        total_steps = len(transitions)
        print('Total Steps: {}'.format(total_steps))
        for idx in range(total_steps // self.args.batch_size):
            indices = np.random.randint(0, total_steps-1, size=self.args.batch_size)
            s_t, x_t_next, a_t, r_t = [], [], [], []
            for t in indices:
                # Get one sample from this episode
                while transitions[t].nonterminal is False:
                    t = np.random.randint(0, total_steps-1)
                # s_t = Framestacked state at timestep t
                framestacked_transition = get_framestacked_transition(t, transitions)
                s_t.append(torch.stack([trans.state for trans in framestacked_transition]))
                x_t_next.append(transitions[t + 1].state)
                a_t.append(framestacked_transition[-1].action)
                r_t.append(framestacked_transition[-1].reward)

            yield torch.stack(s_t).float().to(self.device) / 255., torch.stack(x_t_next).float().to(self.device) / 255., \
                  F.one_hot(torch.tensor(a_t), num_classes=self.num_actions).float().to(self.device), \
                  (torch.tensor(r_t) + 1).to(self.device)

    def do_one_epoch(self, epoch, episodes, log=True, log_epoch=None):
        data_generator = self.generate_batch(episodes)
        epoch_loss, epoch_sd_loss, epoch_reward_loss, steps = 0., 0., 0., 0,
        rew_acc = 0.
        pos_rew_tp, pos_rew_tn, pos_rew_fp, pos_rew_fn = 0, 0, 0, 0
        zero_rew_tp, zero_rew_tn, zero_rew_fp, zero_rew_fn = 0, 0, 0, 0
        for s_t, x_t_next, a_t, r_t in data_generator:
            s_t = s_t.view(self.args.batch_size * 4, 1, s_t.shape[-2], s_t.shape[-1])
            with torch.no_grad():
                f_t, f_t_next = self.encoder(s_t), self.encoder(x_t_next)
                f_t = f_t.view(self.args.batch_size, 4, -1)
            f_t_last = f_t[:, -1, :]
            f_t = f_t.view(self.args.batch_size, -1)
            hiddens = F.relu(self.hidden(torch.bmm(f_t.unsqueeze(2), a_t.unsqueeze(1)).view(self.args.batch_size, -1)))
            sd_predictions = self.sd_predictor(hiddens)
            reward_predictions = F.log_softmax(self.reward_predictor(hiddens), dim=-1)
            # predict |s_{t+1} - s_t| instead of s_{t+1} directly
            sd_loss = F.mse_loss(sd_predictions, f_t_next - f_t_last)
            if r_t.max() == 2:
                reward_loss = F.nll_loss(reward_predictions, r_t, weight=self.class_weights)
            else:
                # If the batch contains no pos. reward, normalize manually
                reward_loss = F.nll_loss(reward_predictions, r_t, weight=self.class_weights, reduction='none')
                reward_loss = reward_loss.sum() / (self.class_weights[r_t].sum() + self.class_weights[2])

            loss = self.args.sd_loss_coeff * sd_loss + reward_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_sd_loss += self.args.sd_loss_coeff * sd_loss.detach().item()
            epoch_reward_loss += reward_loss.detach().item()
            epoch_loss += loss.detach().item()
            steps += 1

            reward_predictions = reward_predictions.argmax(axis=-1)
            rew_acc += (reward_predictions == r_t).float().mean()
            pos_rew_tp += ((reward_predictions == 2)*(r_t == 2)).float().sum()
            pos_rew_fp += ((reward_predictions == 2)*(r_t != 2)).float().sum()
            pos_rew_tn += ((reward_predictions != 2)*(r_t != 2)).float().sum()
            pos_rew_fn += ((reward_predictions != 2)*(r_t == 2)).float().sum()

            zero_rew_tp += ((reward_predictions == 1)*(r_t == 1)).float().sum()
            zero_rew_fp += ((reward_predictions == 1)*(r_t != 1)).float().sum()
            zero_rew_tn += ((reward_predictions != 1)*(r_t != 1)).float().sum()
            zero_rew_fn += ((reward_predictions != 1)*(r_t == 1)).float().sum()

        pos_recall = pos_rew_tp/(pos_rew_fn + pos_rew_tp)
        pos_precision = pos_rew_tp/(pos_rew_tp + pos_rew_fp)

        zero_recall = zero_rew_tp/(zero_rew_fn + zero_rew_tp)
        zero_precision = zero_rew_tp/(zero_rew_tp + zero_rew_fp)

        if log_epoch is None:
            log_epoch = epoch

        self.log_metrics(log_epoch,
                         epoch_loss / steps,
                         epoch_sd_loss / steps,
                         epoch_reward_loss / steps,
                         rew_acc / steps,
                         pos_recall,
                         pos_precision,
                         zero_recall,
                         zero_precision,
                         log)

    def train(self,
              real_transitions,
              init_epoch=0,
              log_last_only=False,
              log_epoch=None):
        self.class_weights = self.generate_reward_class_weights(real_transitions)
        for e in range(init_epoch, init_epoch + self.args.epochs):
            log = e == init_epoch + self.args.epochs - 1 or not log_last_only
            self.do_one_epoch(e, real_transitions, log, log_epoch)

    def predict(self, z, a):
        N = z.size(0)
        hidden = F.relu(self.hidden(
            torch.bmm(z.unsqueeze(2), a.unsqueeze(1)).view(N, -1)))  # outer-product / bilinear integration, then flatten
        z_last = z.view(N, 4, -1)[:, -1, :]  # choose the last latent vector from z
        return z_last + self.sd_predictor(hidden), self.reward_predictor(hidden).argmax(-1) - 1

    def log_metrics(self,
                    epoch_idx,
                    epoch_loss,
                    sd_loss,
                    reward_loss,
                    rew_acc,
                    pos_recall,
                    pos_prec,
                    zero_recall,
                    zero_prec,
                    log):
        print("Epoch: {}, Epoch Loss: {}, SD Loss: {}, Reward Loss: {}, Reward Accuracy: {}".
              format(epoch_idx, epoch_loss, sd_loss, reward_loss, rew_acc))
        print("Pos. Rew. Recall: {:.3f}, Pos. Rew. Prec.: {:.3f}, Zero Rew. Recall: {:.3f}, Zero Rew. Prec.: {:.3f}".format(pos_recall,
                                                                                                                            pos_prec,
                                                                                                                            zero_recall,
                                                                                                                            zero_prec))
        
        if log:
            wandb.log({'Dynamics loss': epoch_loss,
                       'SD Loss': sd_loss,
                       'Reward Loss': reward_loss,
                       "Reward Accuracy": rew_acc,
                       "Pos. Reward Recall": pos_recall,
                       "Zero Reward Recall": zero_recall,
                       "Pos. Reward Precision": pos_prec,
                       "Zero Reward Precision": zero_prec},
                      step=epoch_idx)
