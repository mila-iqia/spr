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
            nn.Linear(input_size+action_size, hidden_size),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x, action):
        return self.network(x, action)


class ForwardModel(object):
    def __int__(self, args, encoder, device):
        self.args = args
        self.device = device
        self.encoder = encoder
        self.model = MLP(args.feature_size, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters())

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
            x_t, x_t_next, a_t = [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t = np.random.randint(0, len(episode) - 1)
                x_t.append(episode[t][0])
                a_t.append(episode[t][1])
                x_t_next.append(episode[t + 1][0])
            yield torch.stack(x_t).to(self.device) / 255., torch.stack(x_t_next).to(self.device) / 255., torch.stack(a_t)

    def do_one_epoch(self, epoch, episodes):
        data_generator = self.generate_batch(episodes)
        epoch_loss, steps = 0., 0
        for x_t, x_t_next, a_t in data_generator:
            with torch.no_grad:
                f_t, f_t_next = self.encoder(x_t), self.encoder(x_t_next)
            predictions = self.model(f_t, a_t)
            loss = F.mse_loss(predictions, f_t_next - f_t) # predict |s_{t+1} - s_t| instead of s_{t+1} directly

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.detach.item()
            steps += 1
        self.log_metrics(epoch, epoch_loss / steps)

    def train(self, train_eps):
        for e in range(self.args.epochs):
            self.do_one_epoch(e, train_eps)

    def predict(self, z, a):
        pass

    def log_metrics(self, epoch_idx, epoch_loss):
        print("Epoch: {}, Epoch Loss: {}".format(epoch_idx, epoch_loss))
        wandb.log({'Dynamics loss': epoch_loss}, step=epoch_idx)
