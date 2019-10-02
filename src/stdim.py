import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
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


class InfoNCESpatioTemporalTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.classifier1 = nn.Linear(self.encoder.hidden_size, 64).to(device)  # x1 = global, x2=patch, n_channels = 32
        self.classifier2 = nn.Linear(128, 128).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.classifier1.parameters()) + list(self.encoder.parameters()) +
                                          list(self.classifier2.parameters()),
                                          lr=config['encoder_lr'], eps=1e-5)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")
        self.epochs_till_now = 0

    def generate_batch(self, transitions):
        total_steps = len(transitions)
        print('Total Steps: {}'.format(len(transitions)))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        for idx in range(total_steps // self.batch_size):
            indices = np.random.randint(0, total_steps-1, size=self.batch_size)
            x_t, x_tnext = [], []
            for t in indices:
                # Get one sample from this episode
                while transitions[t].nonterminal is False:
                    t = np.random.randint(0, total_steps-1)
                x_t.append(transitions[t].state)
                x_tnext.append(transitions[t+1].state)
            yield torch.stack(x_t).to(self.device).float() / 255., torch.stack(x_tnext).to(self.device).float() / 255.

    def nce_with_negs_from_same_loc(self, f_glb, f_lcl):
        '''
        Compute InfoNCE cost with source features in f_glb and target features in
        f_lcl. We assume one source feature vector per item in batch and n_locs
        target feature vectors per item in batch. There are n_batch items in the
        batch and the dimension of source/target feature vectors is n_rkhs.
        -- defn: we condition on source features and predict target features

        For the positive nce pair (f_glb[i, :], f_lcl[i, :, l]), which comes from
        batch item i at spatial location l, we will use the target feature vectors
        f_lcl[j, :, l] as negative samples, for all j != i.

        Input:
          f_glb : (n_batch, n_rkhs)          -- one source vector per item
          f_lcl : (n_batch, n_rkhs, n_locs)  -- n_locs target vectors per item
        Output:
          loss_nce : (n_batch, n_locs)       -- InfoNCE cost at each location
        '''
        n_batch = f_lcl.size(0)
        n_batch_glb = f_glb.size(0)
        n_rkhs = f_glb.size(1)
        n_locs = f_lcl.size(2)
        # reshaping for big matrix multiply
        f_glb = f_glb.permute(1, 0)  # (n_rkhs, n_batch)
        f_lcl = f_lcl.permute(0, 2, 1)  # (n_batch, n_locs, n_rkhs)
        f_lcl = f_lcl.reshape(n_batch * n_locs, n_rkhs)  # (n_batch*n_locs, n_rkhs)
        # compute raw scores dot(f_glb[i, :], f_lcl[j, :, l]) for all i, j, l
        raw_scores = torch.mm(f_lcl, f_glb)  # (n_batch*n_locs, n_batch)
        raw_scores = raw_scores.reshape(n_batch, n_locs, n_batch_glb)  # (n_batch, n_locs, n_batch)
        # now, raw_scores[j, l, i] = dot(f_glb[i, :], f_lcl[j, :, l])
        # -- we can get NCE log softmax by normalizing over the j dimension...
        nce_lsmax = -F.log_softmax(raw_scores, dim=0)  # (n_batch, n_locs, n_batch)
        # make a mask for picking out the log softmax values for positive pairs
        pos_mask = torch.eye(n_batch, dtype=nce_lsmax.dtype, device=nce_lsmax.device)
        if pos_mask.shape[-1] != raw_scores.shape[-1]:
            new_zeros = torch.zeros(n_batch,
                                    raw_scores.shape[-1] - pos_mask.shape[-1],
                                    device=nce_lsmax.device,
                                    dtype=nce_lsmax.dtype)
            pos_mask = torch.cat([pos_mask, new_zeros], -1)
        pos_mask = pos_mask.unsqueeze(dim=1)
        # use a masked sum over the j dimension to select positive pair NCE scores
        loss_nce = (nce_lsmax * pos_mask).sum(dim=0)  # (n_locs, n_batch)
        # permute axes to make return shape consistent with input shape
        loss_nce = loss_nce.permute(1, 0)  # (n_batch, n_locs)
        return loss_nce

    def do_one_epoch(self, episodes):
        mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        accuracy1, accuracy2 = 0., 0.
        epoch_loss1, epoch_loss2 = 0., 0.
        data_generator = self.generate_batch(episodes)
        for x_t, x_tnext in data_generator:
            f_t_maps, f_t_next_maps = self.encoder(x_t, fmaps=True), self.encoder(x_tnext, fmaps=True)

            # Loss 1: Global at time t, f5 patches at time t-1
            f_t, f_t_next = f_t_maps['out'], f_t_next_maps['f5']
            predictions = self.classifier1(f_t)
            f_t_next = f_t_next.flatten(1, 2).transpose(-1, -2)
            loss = self.nce_with_negs_from_same_loc(predictions, f_t_next).mean()

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            steps += 1
        self.log_results(epoch_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    def train(self, tr_eps, val_eps=None, epochs=None):
        if not epochs:
            epochs = self.epochs
        for e in range(epochs):
            self.encoder.train(), self.classifier1.train(), self.classifier2.train()
            self.do_one_epoch(tr_eps)

            if val_eps:
                self.encoder.eval(), self.classifier1.eval(), self.classifier2.eval()
                self.do_one_epoch(val_eps)

                if self.early_stopper.early_stop:
                    break
            self.epochs_till_now += 1
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['game'] + '.pt'))

    def log_results(self, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {}".format(prefix.capitalize(), self.epochs_till_now, epoch_loss,
                                                                     prefix.capitalize()))
        self.wandb.log({prefix + '_loss': epoch_loss, 'STDIM-epoch': self.epochs_till_now})
