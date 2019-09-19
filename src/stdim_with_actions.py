import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout
from .trainer import Trainer
from src.utils import EarlyStopping
from torchvision import transforms
import torchvision.transforms.functional as TF


class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class PredictionModule(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, state_dim+action_dim),
            nn.ReLU(),
            nn.Linear(state_dim+action_dim, state_dim))

    def forward(self, states, actions):
        inp = torch.cat([states, actions], -1)
        return states + self.network(inp)


class GlobalLocalInfoNCESpatioTemporalTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.use_multiple_predictors = config.get("use_multiple_predictors", False)
        print("Using multiple predictors" if self.use_multiple_predictors else "Using shared classifier")
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.classifier = nn.Linear(self.encoder.hidden_size, 128).to(device)
        self.params = list(self.encoder.parameters())
        self.params += list(self.classifier.parameters())

        self.action_embedding = nn.Embedding(config["num_actions"],
                                             config["action_embed_dim"])
        self.prediction_module = PredictionModule(self.encoder.hidden_size,
                                                  config["action_embed_dim"])
        self.action_embedding.to(device)
        self.prediction_module.to(device)
        self.params += list(self.action_embedding.parameters())
        self.params += list(self.prediction_module.parameters())
        self.hard_neg_factor = config["hard_neg_factor"]

        self.optimizer = torch.optim.Adam(self.params, lr=config['lr'], eps=1e-5)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")



    def generate_batch(self, transitions, actions=None):
        total_steps = len(transitions)
        print('Total Steps: {}'.format(len(transitions)))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        for idx in range(total_steps // self.batch_size):
            indices = np.random.randint(0, total_steps-1, size=self.batch_size)
            x_t, x_tnext, a_t = [], [], []
            if actions is not None:
                a = []
            for t in indices:
                # Get one sample from this episode
                while transitions[t].nonterminal is False:
                    t = np.random.randint(0, total_steps-1)
                x_t.append(transitions[t].state)
                x_tnext.append(transitions[t+1].state)
                a_t.append(transitions[t].action)

            yield torch.stack(x_t).to(self.device).float() / 255., \
                  torch.stack(x_tnext).to(self.device).float() / 255.,\
                  torch.stack(a_t).to(self.device).long()

    def hard_neg_sampling(self, encoded_obs, emb_actions):
        """
        :param encoded_obs: Output of the encoder.
        :param emb_actions: Embedded (or otherwise continuous) actions
        :return: A tensor of predicted states for negative pairs.
        """

        encoded_obs = encoded_obs.repeat(self.hard_neg_factor, 1)
        actions = emb_actions.repeat(self.hard_neg_factor, 1)

        shuffle_indices = torch.randperm(actions.shape[0],
                                         device=actions.device)
        shuffled_actions = actions[shuffle_indices]

        predictions = self.prediction_module(encoded_obs, shuffled_actions)
        return predictions

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        epoch_loss1, epoch_loss2 = 0., 0.
        data_generator = self.generate_batch(episodes)
        for x_t, x_tprev, actions in data_generator:
            f_t_maps, f_t_prev_maps = self.encoder(x_t, fmaps=True),\
                                      self.encoder(x_tprev, fmaps=True)

            # Loss 1: Global at time t, f5 patches at time t-1
            f_t, f_t_prev = f_t_maps['f5'], f_t_prev_maps['out']
            sy = f_t.size(1)
            sx = f_t.size(2)

            N = f_t_prev.size(0)
            loss1 = 0.

            embedded_actions = self.action_embedding(actions).squeeze(1)
            f_t_pred = self.prediction_module(f_t_prev, embedded_actions)
            if self.hard_neg_factor > 0:
                hard_negs = self.hard_neg_sampling(f_t_prev,
                                                   embedded_actions)
                f_t_pred = torch.cat([f_t_pred, hard_negs], 0)

            for y in range(sy):
                for x in range(sx):
                    predictions = self.classifier1(f_t_pred)
                    positive = f_t[:, y, x, :]
                    logits = torch.matmul(predictions, positive.t())
                    step_loss = F.cross_entropy(logits.t(), torch.arange(N).to(self.device))
                    loss1 += step_loss
            loss1 = loss1 / (sx * sy)

            self.optimizer.zero_grad()
            loss = loss1
            if mode == "train":
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_loss1 += loss1.detach().item()
            steps += 1
        self.log_results(epoch, epoch_loss1 / steps, epoch_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    def train(self, tr_eps, val_eps=None):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        for e in range(self.epochs):
            self.encoder.train(), self.classifier1.train()
            self.do_one_epoch(e, tr_eps)

            if val_eps:
                self.encoder.eval(), self.classifier1.eval()
                self.do_one_epoch(e, val_eps)

                if self.early_stopper.early_stop:
                    break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['game'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss1, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                     prefix.capitalize()))
        self.wandb.log({prefix + '_loss': epoch_loss,
                        prefix + '_loss1': epoch_loss1}, step=epoch_idx)
