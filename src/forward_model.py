import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.data import RandomSampler, BatchSampler
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size=256, action_size=1, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size + action_size, hidden_size),
            nn.ReLU()
        )
        self.state_diff = nn.Linear(hidden_size, input_size)
        self.reward_predictor = nn.Linear(hidden_size, 1)

    def forward(self, x, action):
        return self.network(torch.cat((x, action), dim=-1))


class ForwardModel():
    def __init__(self, args, encoder, device):
        self.args = args
        self.device = device
        self.encoder = encoder
        hidden_size = args.forward_hidden_size
        self.hidden = nn.Linear(args.feature_size + 1, hidden_size)
        self.sd_predictor = nn.Linear(hidden_size, args.feature_size)
        self.reward_predictor = nn.Linear(hidden_size, 1)
        self.optimizer = torch.optim.Adam(list(self.hidden.parameters()) + list(self.sd_predictor.parameters()) +
                                          self.reward_predictor.parameters())

    def generate_batch(self, episodes):
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.args.batch_size, drop_last=True)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            x_t, x_t_next, a_t, r_t = [], [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t = np.random.randint(0, len(episode) - 1)
                x_t.append(episode[t][0])
                a_t.append(episode[t][1])
                r_t.append(episode[t][2])
                x_t_next.append(episode[t + 1][0])
            yield torch.stack(x_t).to(self.device) / 255., torch.stack(x_t_next).to(self.device) / 255., \
                  torch.stack(a_t).float(), torch.stack(r_t)

    def do_one_epoch(self, epoch, episodes):
        data_generator = self.generate_batch(episodes)
        epoch_loss, epoch_sd_loss, epoch_reward_loss, steps = 0., 0., 0., 0
        for x_t, x_t_next, a_t, r_t in data_generator:
            with torch.no_grad():
                f_t, f_t_next = self.encoder(x_t), self.encoder(x_t_next)
            hiddens = self.hidden(f_t, a_t)
            sd_predictions = self.sd_predictor(hiddens)
            reward_predictions = self.reward_predictor(hiddens)
            # predict |s_{t+1} - s_t| instead of s_{t+1} directly
            sd_loss = F.mse_loss(sd_predictions, f_t_next - f_t)
            reward_loss = F.mse_loss(reward_predictions, r_t)

            loss = sd_loss + reward_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_sd_loss += sd_loss.detach().item()
            epoch_reward_loss += reward_loss.detach().item()
            epoch_loss += loss.detach().item()
            steps += 1
        self.log_metrics(epoch, epoch_loss / steps, epoch_reward_loss / steps, epoch_reward_loss / steps)

    def train(self, train_eps):
        for e in range(self.args.epochs):
            self.do_one_epoch(e, train_eps)

    def predict(self, z, a):
        hidden = self.hidden(z, a)
        return self.sd_predictor(hidden), self.reward_predictor(hidden)

    def log_metrics(self, epoch_idx, epoch_loss, sd_loss, reward_loss):
        print("Epoch: {}, Epoch Loss: {}, SD Loss: {}, Reward Loss: {}".
              format(epoch_idx, epoch_loss, sd_loss, reward_loss))
        wandb.log({'Dynamics loss': epoch_loss,
                   'SD Loss': sd_loss,
                   'Reward Loss': reward_loss}, step=epoch_idx)
