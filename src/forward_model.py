import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.data import RandomSampler, BatchSampler
import torch.nn.functional as F

from src.episodes import get_framestacked_transition


class ForwardModel():
    def __init__(self, args, encoder, num_actions):
        self.args = args
        self.device = args.device
        self.encoder = encoder
        self.num_actions = num_actions
        hidden_size = args.forward_hidden_size
        self.hidden = nn.Linear(args.feature_size * 4 * num_actions, hidden_size).to(self.device)
        self.sd_predictor = nn.Linear(hidden_size, args.feature_size).to(self.device)
        self.reward_predictor = nn.Linear(hidden_size, 1).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.hidden.parameters()) + list(self.sd_predictor.parameters()) +
                                          list(self.reward_predictor.parameters()))

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
                  torch.tensor(r_t).unsqueeze(-1).float().to(self.device)

    def do_one_epoch(self, epoch, episodes):
        data_generator = self.generate_batch(episodes)
        epoch_loss, epoch_sd_loss, epoch_reward_loss, steps = 0., 0., 0., 0
        for s_t, x_t_next, a_t, r_t in data_generator:
            s_t = s_t.view(self.args.batch_size * 4, 1, s_t.shape[-2], s_t.shape[-1])
            with torch.no_grad():
                f_t, f_t_next = self.encoder(s_t), self.encoder(x_t_next)
                f_t = f_t.view(self.args.batch_size, 4, -1)
            f_t_last = f_t[:, -1, :]
            f_t = f_t.view(self.args.batch_size, -1)
            hiddens = F.relu(self.hidden(torch.bmm(f_t.unsqueeze(2), a_t.unsqueeze(1)).view(self.args.batch_size, -1)))
            sd_predictions = self.sd_predictor(hiddens)
            reward_predictions = self.reward_predictor(hiddens)
            # predict |s_{t+1} - s_t| instead of s_{t+1} directly
            sd_loss = F.mse_loss(sd_predictions, f_t_next - f_t_last)
            reward_loss = F.mse_loss(reward_predictions, r_t)

            loss = sd_loss + reward_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_sd_loss += sd_loss.detach().item()
            epoch_reward_loss += reward_loss.detach().item()
            epoch_loss += loss.detach().item()
            steps += 1
        self.log_metrics(epoch, epoch_loss / steps, epoch_sd_loss / steps, epoch_reward_loss / steps)

    def train(self, real_transitions):
        for e in range(self.args.epochs):
            self.do_one_epoch(e, real_transitions)

    def predict(self, z, a):
        N = z.size(0)
        hidden = F.relu(self.hidden(
            torch.bmm(z.unsqueeze(2), a.unsqueeze(1)).view(N, -1)))  # outer-product / bilinear integration, then flatten
        z_last = z.view(N, 4, -1)[:, -1, :]  # choose the last latent vector from z
        return z_last + self.sd_predictor(hidden), self.reward_predictor(hidden)

    def log_metrics(self, epoch_idx, epoch_loss, sd_loss, reward_loss):
        print("Epoch: {}, Epoch Loss: {}, SD Loss: {}, Reward Loss: {}".
              format(epoch_idx, epoch_loss, sd_loss, reward_loss))
        wandb.log({'Dynamics loss': epoch_loss,
                   'SD Loss': sd_loss,
                   'Reward Loss': reward_loss})
