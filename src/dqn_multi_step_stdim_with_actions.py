import random

import torch
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout
from .trainer import Trainer
from src.utils import EarlyStopping, fig2data, save_to_pil
from torchvision import transforms
import torchvision.transforms.functional as TF
from src.memory import blank_trans
from torch.distributions.categorical import Categorical
import wandb


class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class FILM(nn.Module):
    def __init__(self, input_dim, cond_dim, layernorm=True):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.conditioning = nn.Linear(cond_dim, input_dim*2)
        self.layernorm = layernorm

    def forward(self, input, cond):
        conditioning = self.conditioning(cond)
        gamma = conditioning[..., :self.input_dim]
        beta = conditioning[..., self.input_dim:]
        if self.layernorm:
            input = F.layer_norm(input, input.shape[1:])

        while len(input.shape) > len(beta.shape):
            beta = beta.unsqueeze(-1)
            gamma = gamma.unsqueeze(-1)

        return input*gamma + beta


class ConvolutionalPredictionModule(nn.Module):
    def __init__(self, state_dim, num_actions, layernorm=False, layers=3,
                 h_size=-1):
        super().__init__()
        if h_size <= 0:
            h_size = 4*state_dim
        self.convert_actions = lambda a: F.one_hot(a, num_classes=num_actions)
        self.films = nn.ModuleList()
        self.films.append(FILM(state_dim*4, num_actions, layernorm=layernorm))
        for layer in range(layers - 1):
            self.films.append(FILM(h_size, num_actions, layernorm=layernorm))

        network_layers = [nn.Conv2d(4*state_dim, h_size, kernel_size=3,
                                    padding=1),
                          nn.ReLU()]
        for i in range(layers - 2):
            network_layers.append(nn.Conv2d(h_size, h_size, kernel_size=3,
                                            padding=1))
            network_layers.append(nn.ReLU())
        network_layers.append(nn.Conv2d(h_size, state_dim, kernel_size=3,
                                        padding=1))
        self.network = nn.Sequential(*network_layers)

    def forward(self, states, actions):
        actions = self.convert_actions(actions).float()
        current = states
        for i, film in enumerate(self.films):
            current = film(current, actions)
            current = self.network[i*2:i*2+2](current)
        return current


class FILMPredictionModule(nn.Module):
    def __init__(self, state_dim, num_actions, layernorm=False, layers=3,
                 h_size=-1):
        super().__init__()
        if h_size <= 0:
            h_size = 4*state_dim
        self.convert_actions = lambda a: F.one_hot(a, num_classes=num_actions)
        self.films = nn.ModuleList()
        self.films.append(FILM(state_dim*4, num_actions, layernorm=layernorm))
        for layer in range(layers - 1):
            self.films.append(FILM(h_size, num_actions, layernorm=layernorm))

        network_layers = [nn.Linear(4*state_dim, h_size),
                          nn.ReLU()]
        for i in range(layers - 2):
            network_layers.append(nn.Linear(h_size, h_size))
            network_layers.append(nn.ReLU())
        network_layers.append(nn.Linear(h_size, state_dim))
        self.network = nn.Sequential(*network_layers)

    def forward(self, states, actions):
        actions = self.convert_actions(actions).float()
        current = states
        for i, film in enumerate(self.films):
            current = film(current, actions)
            current = self.network[i*2:i*2+2](current)
        return current


class FILMRewardPredictionModule(nn.Module):
    def __init__(self, state_dim, num_actions, reward_dim=3,
                 layernorm=False, layers=3,
                 h_size=-1, dropout=0):
        super().__init__()
        if h_size <= 0:
            h_size = 4*state_dim
        self.convert_actions = lambda a: F.one_hot(a, num_classes=num_actions)
        self.films = nn.ModuleList()
        self.films.append(FILM(state_dim*4, num_actions, layernorm=layernorm))
        for layer in range(layers - 1):
            self.films.append(FILM(h_size, num_actions, layernorm=layernorm))

        network_layers = [nn.Linear(4*state_dim, h_size),
                          nn.ReLU(),
                          nn.Dropout(dropout)]
        for i in range(layers - 2):
            network_layers.append(nn.Linear(h_size, h_size))
            network_layers.append(nn.ReLU())
            network_layers.append(nn.Dropout(dropout))
        network_layers.append(nn.Linear(h_size, reward_dim))
        self.network = nn.Sequential(*network_layers)

    def forward(self, states, actions):
        actions = self.convert_actions(actions).float()
        current = states
        for i, film in enumerate(self.films):
            current = film(current, actions)
            current = self.network[i*3:i*3+3](current)
        return current


class PredictionModule(nn.Module):
    def __init__(self, state_dim, num_actions, layers=3,
                 h_size=-1):
        super().__init__()
        if h_size <= 0:
            h_size = 4*state_dim
        self.convert_actions = lambda a: F.one_hot(a, num_classes=num_actions)
        if layers == 1:
            network_layers = [nn.Linear(state_dim*4*num_actions, state_dim)]
        else:
            network_layers = [nn.Linear(state_dim*4*num_actions, h_size),
                              nn.ReLU()]
            for i in range(layers - 2):
                network_layers.append(nn.Linear(h_size, h_size))
                network_layers.append(nn.ReLU())
            network_layers.append(nn.Linear(h_size, state_dim))
        self.network = nn.Sequential(*network_layers)

    def forward(self, states, actions):
        actions = self.convert_actions(actions).float()
        N = states.size(0)
        output = self.network(
            torch.bmm(states.unsqueeze(2), actions.unsqueeze(1)).view(N, -1))  # outer-product / bilinear integration, then flatten
        return output


class RewardPredictionModule(nn.Module):
    def __init__(self, state_dim, num_actions, reward_dim=3, layers=3,
                 h_size=-1, dropout=0):
        super().__init__()
        if h_size <= 0:
            h_size = 4*state_dim
        self.convert_actions = lambda a: F.one_hot(a, num_classes=num_actions)
        if layers == 1:
            network_layers = [nn.Linear(h_size*num_actions, reward_dim)]
        else:
            network_layers = [nn.Linear(state_dim*4*num_actions, h_size),
                              nn.ReLU(),
                              nn.Dropout(dropout)]
            for i in range(layers - 2):
                network_layers.append(nn.Linear(h_size, h_size))
                network_layers.append(nn.ReLU())
                network_layers.append(nn.Dropout(dropout))
            network_layers.append(nn.Linear(h_size, reward_dim))
        self.network = nn.Sequential(*network_layers)

    def forward(self, states, actions):
        actions = self.convert_actions(actions).float()
        N = states.size(0)
        output = self.network(
            torch.bmm(states.unsqueeze(2), actions.unsqueeze(1)).view(N, -1))  # outer-product / bilinear integration, then flatten
        return output


class MultiStepActionInfoNCESpatioTemporalTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None, agent=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.use_multiple_predictors = config.get("use_multiple_predictors", False)
        print("Using multiple predictors" if self.use_multiple_predictors else "Using shared classifier")
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.global_loss = config['global_loss']
        self.local_loss = config['local_loss']
        self.bilinear_global_loss = config['bilinear_global_loss']
        self.noncontrastive_global_loss = config['noncontrastive_global_loss']
        self.noncontrastive_loss_weight = config['noncontrastive_loss_weight']

        self.agent = agent
        self.online_agent_training = config["online_agent_training"]

        self.device = device
        self.classifier = nn.Linear(self.encoder.hidden_size, self.encoder.f5_size).to(device)
        self.global_classifier = nn.Linear(self.encoder.hidden_size,
                                           self.encoder.hidden_size).to(device)
        self.params = list(self.encoder.parameters())
        self.params += list(self.classifier.parameters())
        self.params += list(self.global_classifier.parameters())

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_update = self.config["target_update"]
        self.steps = 0

        if self.local_loss:
            self.local_prediction_module = ConvolutionalPredictionModule(
                self.encoder.f5_size,
                config["num_actions"],
                layernorm=config["layernorm"],
                layers=config["prediction_layers"],
                h_size=config["prediction_hidden"]
            )
            self.local_classifier = nn.Linear(self.encoder.f5_size,
                                              self.encoder.f5_size).to(device)
            self.params += list(self.local_prediction_module.parameters())
            self.params += list(self.local_classifier.parameters())
            self.local_prediction_module.to(device)
            self.local_classifier.to(device)

        if config["film"]:
            self.prediction_module = FILMPredictionModule(self.encoder.hidden_size,
                                                          config["num_actions"],
                                                          layernorm=config["layernorm"],
                                                          layers=config["prediction_layers"],
                                                          h_size=config["prediction_hidden"])
            self.reward_module = FILMRewardPredictionModule(self.encoder.hidden_size,
                                                            config["num_actions"],
                                                            layernorm=config["layernorm"],
                                                            layers=config["reward_layers"],
                                                            h_size=config["reward_hidden"],
                                                            dropout=config["dropout_prob"])
            self.done_module = FILMRewardPredictionModule(self.encoder.hidden_size,
                                                          config["num_actions"],
                                                          layernorm=config["layernorm"],
                                                          layers=config["reward_layers"],
                                                          h_size=config["reward_hidden"],
                                                          dropout=config["dropout_prob"],
                                                          reward_dim=2)

        else:
            self.prediction_module = PredictionModule(self.encoder.hidden_size,
                                                      layers=config["prediction_layers"],
                                                      h_size=config["prediction_hidden"],
                                                      num_actions=config["num_actions"])
            self.reward_module = RewardPredictionModule(self.encoder.hidden_size,
                                                        config["num_actions"],
                                                        layers=config["reward_layers"],
                                                        h_size=config["reward_hidden"],
                                                        dropout=config["dropout_prob"])
            self.done_module = RewardPredictionModule(self.encoder.hidden_size,
                                                      config["num_actions"],
                                                      layers=config["reward_layers"],
                                                      h_size=config["reward_hidden"],
                                                      dropout=config["dropout_prob"],
                                                      reward_dim=2)

        self.reward_loss_weight = config["reward_loss_weight"]

        self.dense_supervision = config["dense_supervision"]

        self.dqn_loss_weight = config["dqn_loss_weight"]

        self.no_class_weighting = config["no_class_weighting"]

        self.prediction_module.to(device)
        self.reward_module.to(device)
        self.done_module.to(device)
        self.params += list(self.prediction_module.parameters())
        self.params += list(self.reward_module.parameters())
        self.params += list(self.done_module.parameters())
        self.hard_neg_factor = config["hard_neg_factor"]

        self.maximum_length = config["max_jump_length"]
        self.minimum_length = config["min_jump_length"]

        self.detach_target = config["detach_target"]

        self.optimizer = torch.optim.Adam(self.params, lr=config['encoder_lr'], eps=1e-5)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="model")
        self.epochs_till_now = 0
        self.train_trackers = dict()
        self.reset_trackers("train")
        self.val_trackers = dict()
        self.reset_trackers("val")

    def reset_es(self):
        self.early_stopper = EarlyStopping(patience=self.patience,
                                           verbose=False,
                                           wandb=self.wandb,
                                           name="model")

    def maybe_update_target_net(self):
        # Check also to see if the agent has updated on its own due to
        # concurrent updates; if so, update as well.
        if self.steps > self.target_update or self.steps > self.agent.steps:
            self.update_target_net()
            self.agent.update_target_net()

    def update_target_net(self):
        self.steps = 0
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    def nce_per_location(self, f_x1, f_x2):
        '''
        Compute InfoNCE cost with source features in f_x1 and target features in
        f_x2. We assume one source feature vector per location per item in batch
        and one target feature vector per location per item in batch. There are
        n_batch items, n_locs locations, and n_rkhs dimensions per vector.
        -- note: we can predict x1->x2 and x2->x1 in parallel "for free"

        For the positive nce pair (f_x1[i, :, l], f_x2[i, :, l]), which comes from
        batch item i at spatial location l, we will use the target feature vectors
        f_x2[j, :, l] as negative samples, for all j != i.

        Input:
          f_x1 : (n_batch, n_rkhs, n_locs)  -- n_locs source vectors per item
          f_x2 : (n_batch, n_rkhs, n_locs)  -- n_locs target vectors per item
        Output:
          loss_nce : (n_batch, n_locs)       -- InfoNCE cost at each location
        '''
        n_batch = f_x1.size(0)
        n_rkhs = f_x1.size(1)
        n_locs = f_x1.size(2)
        # reshaping for big matrix multiply
        f_x1 = f_x1.permute(2, 0, 1)  # (n_locs, n_batch, n_rkhs)
        f_x2 = f_x2.permute(2, 1, 0)  # (n_locs, n_rkhs, n_batch)
        # compute dot(f_glb[i, :, l], f_lcl[j, :, l]) for all i, j, l
        # -- after matmul: raw_scores[l, i, j] = dot(f_x1[i, :, l], f_x2[j, :, l])
        raw_scores = torch.matmul(f_x1, f_x2)  # (n_locs, n_batch, n_batch)
        # We get NCE log softmax by normalizing over dim 1 or 2 of raw_scores...
        # -- normalizing over dim 1 gives scores for predicting x2->x1
        # -- normalizing over dim 2 gives scores for predicting x1->x2
        lsmax_x1_to_x2 = -F.log_softmax(raw_scores, dim=2)  # (n_locs, n_batch, n_batch)
        lsmax_x2_to_x1 = -F.log_softmax(raw_scores, dim=1)  # (n_locs, n_batch, n_batch)
        # make a mask for picking out the NCE scores for positive pairs
        pos_mask = torch.eye(n_batch, dtype=f_x1.dtype, device=f_x1.device)
        pos_mask = pos_mask.unsqueeze(dim=0)
        # use masked sums to select NCE scores for positive pairs
        loss_nce_x1_to_x2 = (lsmax_x1_to_x2 * pos_mask).sum(dim=2)  # (n_locs, n_batch)
        loss_nce_x2_to_x1 = (lsmax_x2_to_x1 * pos_mask).sum(dim=1)  # (n_locs, n_batch)
        # combine forwards/backwards prediction costs (or whatever)
        loss_nce = 0.5 * (loss_nce_x1_to_x2 + loss_nce_x2_to_x1)
        return loss_nce

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

    def generate_reward_class_weights(self, memory, uniform=False):
        if uniform:
            weights = [1., 1., 1.]
            return torch.tensor(weights, device=self.device)

        counts = [0, 0, 0]  # counts for reward=-1,0,1
        for i in range(len(memory.transitions)):
            trans = memory.transitions.get(i)
            counts[trans.reward + 1] += 1

        weights = [0., 0., 0.]
        for i in range(3):
            if counts[i] != 0:
                weights[i] = sum(counts) / counts[i]
        return torch.tensor(weights, device=self.device)

    def update_reward_trackers(self, n, rewards, reward_preds, mode="train"):
        if mode == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers
        trackers["pos_rew_tps"][n] += ((reward_preds == 2)*(rewards == 2)).float().sum()
        trackers["pos_rew_fps"][n] += ((reward_preds == 2)*(rewards != 2)).float().sum()
        trackers["pos_rew_fns"][n] += ((reward_preds != 2)*(rewards == 2)).float().sum()
        trackers["pos_rew_tns"][n] += ((reward_preds != 2)*(rewards != 2)).float().sum()

        trackers["zero_rew_tps"][n] += ((reward_preds == 1)*(rewards == 1)).float().sum()
        trackers["zero_rew_fps"][n] += ((reward_preds == 1)*(rewards != 1)).float().sum()
        trackers["zero_rew_fns"][n] += ((reward_preds != 1)*(rewards == 1)).float().sum()
        trackers["zero_rew_tns"][n] += ((reward_preds != 1)*(rewards != 1)).float().sum()

    def update_done_trackers(self, dones, done_preds, mode="train"):
        if mode == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers
        trackers["done_tps"] += ((done_preds == 0)*(dones == 0)).float().sum()
        trackers["done_fps"] += ((done_preds == 0)*(dones != 0)).float().sum()
        trackers["done_fns"] += ((done_preds != 0)*(dones == 0)).float().sum()
        trackers["done_tns"] += ((done_preds != 0)*(dones != 0)).float().sum()

    def update_cos_sim_trackers(self, n, cos_sim, sd_loss, mode="train"):
        if mode == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers
        trackers["cos_sims"][n] += cos_sim
        trackers["sd_losses"][n] += sd_loss
        trackers["counts"][n] += 1

    def summarize_trackers(self, mode="train"):
        if mode == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers
        pos_recalls = []
        pos_precs = []
        zero_recalls = []
        zero_precs = []
        rew_accs = []
        done_recall = (trackers["done_tps"] / (trackers["done_fns"] + trackers["done_tps"]))
        done_prec = (trackers["done_tps"] / (trackers["done_tps"] + trackers["done_fps"]))
        done_acc = (trackers["done_tps"] + trackers["done_tns"]) / \
                   (trackers["done_fns"] + trackers["done_tps"] +
                    trackers["done_fps"] + trackers["done_tns"])
        for i in range(self.maximum_length):
            pos_recalls.append(trackers["pos_rew_tps"][i] / (trackers["pos_rew_fns"][i] + trackers["pos_rew_tps"][i]))
            pos_precs.append(trackers["pos_rew_tps"][i] / (trackers["pos_rew_tps"][i] + trackers["pos_rew_fps"][i]))

            zero_recalls.append(trackers["zero_rew_tps"][i] / (trackers["zero_rew_fns"][i] + trackers["zero_rew_tps"][i]))
            zero_precs.append(trackers["zero_rew_tps"][i] / (trackers["zero_rew_tps"][i] + trackers["zero_rew_fps"][i]))
            acc = (trackers["pos_rew_tps"][i] + trackers["pos_rew_tns"][i]) / \
                  (trackers["pos_rew_fns"][i] + trackers["pos_rew_tps"][i] +
                   trackers["pos_rew_fps"][i] + trackers["pos_rew_tns"][i])
            rew_accs.append(acc)

        pos_recall = (np.sum(trackers["pos_rew_tps"]) / (np.sum(trackers["pos_rew_fns"]) + np.sum(trackers["pos_rew_tps"])))
        pos_prec = (np.sum(trackers["pos_rew_tps"]) / (np.sum(trackers["pos_rew_tps"]) + np.sum(trackers["pos_rew_fps"])))
        zero_recall = (np.sum(trackers["zero_rew_tps"]) / (np.sum(trackers["zero_rew_fns"]) + np.sum(trackers["zero_rew_tps"])))
        zero_prec = (np.sum(trackers["zero_rew_tps"]) / (np.sum(trackers["zero_rew_tps"]) + np.sum(trackers["zero_rew_fps"])))
        acc = (np.sum(trackers["pos_rew_tps"]) + np.sum(trackers["pos_rew_tns"])) / \
              (np.sum(trackers["pos_rew_fns"]) + np.sum(trackers["pos_rew_tps"]) +
               np.sum(trackers["pos_rew_fps"]) + np.sum(trackers["pos_rew_tns"]))

        cosine_sim = np.sum(trackers["cos_sims"])/np.sum(trackers["counts"])
        cosine_sims = trackers["cos_sims"]/trackers["counts"]
        sd_losses = trackers["sd_losses"]/trackers["counts"]
        sd_loss = np.sum(trackers["sd_losses"]/np.sum(trackers["counts"]))

        return rew_accs, pos_recalls, pos_precs, zero_recalls, zero_precs,\
               acc, pos_recall, pos_prec, zero_recall, zero_prec, \
               cosine_sims, cosine_sim, sd_losses, sd_loss, \
               done_acc, done_recall, done_prec

    def reset_trackers(self, mode="train"):
        if mode == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers
        trackers["epoch_rew_loss"] = 0
        trackers["epoch_done_loss"] = 0
        trackers["epoch_global_loss"] = 0
        trackers["epoch_local_local_loss"] = 0
        trackers["rew_acc"] = 0
        trackers["epoch_loss"] = 0
        trackers["online_dqn_loss"] = 0
        trackers["true_representation_norm"] = 0
        trackers["pred_representation_norm"] = 0
        trackers["iterations"] = 0
        trackers["jumps"] = 0
        trackers["sd_loss"] = 0
        trackers["epoch_local_loss"] = 0.
        trackers["cos_sims"] = np.zeros(self.maximum_length)
        trackers["counts"] = np.zeros(self.maximum_length)
        trackers["sd_losses"] = np.zeros(self.maximum_length)

        trackers["pos_rew_tps"] = np.zeros(self.maximum_length)
        trackers["pos_rew_fps"] = np.zeros(self.maximum_length)
        trackers["pos_rew_fns"] = np.zeros(self.maximum_length)
        trackers["pos_rew_tns"] = np.zeros(self.maximum_length)

        trackers["zero_rew_tps"] = np.zeros(self.maximum_length)
        trackers["zero_rew_fps"] = np.zeros(self.maximum_length)
        trackers["zero_rew_fns"] = np.zeros(self.maximum_length)
        trackers["zero_rew_tns"] = np.zeros(self.maximum_length)

        trackers["done_tps"] = 0
        trackers["done_fps"] = 0
        trackers["done_fns"] = 0
        trackers["done_tns"] = 0

    def do_one_epoch(self, memory,
                     plots=False,
                     iterations=-1,
                     log=False):
        mode = "train" if self.encoder.training else "val"

        if mode == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers

        iter_loss = 0
        if iterations <= 0:
            iterations = len(memory.transitions)//self.batch_size
        trackers["iterations"] += iterations
        for _ in range(iterations):
            idxs, x_tprev,\
            actions, rewards, x_t, all_states, nonterminals,\
            weights, n = memory.sample_multistep(self.batch_size,
                                                 self.minimum_length,
                                                 self.maximum_length)
            x_tprev = x_tprev.float().unsqueeze(2)/255.
            x_t = x_t.float().unsqueeze(2)/255.
            all_states = all_states.float().unsqueeze(2)/255.
            rewards = rewards + 1

            loss = self.do_iteration(x_tprev,
                                     x_t,
                                     actions,
                                     rewards,
                                     nonterminals,
                                     all_states,
                                     n,
                                     weights,
                                     mode=mode)
            iter_loss += loss.mean().item()
            if mode == "train":
                memory.update_priorities(idxs, loss.detach().cpu().numpy())
        if mode == "val":
            self.early_stopper(-trackers["epoch_loss"] / trackers["iterations"],
                               self)
        if log:
            self.log_results(mode, plots)

        return iter_loss / iterations

    def do_iteration(self, x_tprev, x_t, actions, rewards, done, all_states, n,
                     weights=1,
                     mode="train"):
        if mode == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers

        init_shape = x_t.shape
        x_tprev = x_tprev.view(init_shape[0]*4, *init_shape[2:])
        x_t = x_t.view(init_shape[0]*4, *init_shape[2:])
        f_t_maps, f_t_prev_maps = self.encoder(x_t, fmaps=True),\
                                  self.encoder(x_tprev, fmaps=True)

        f_t_prev_stack = f_t_prev_maps["out"].view(init_shape[0], 4, -1)
        f_t_initial = f_t_prev_stack[:, -1]
        f_t_prev = f_t_prev_maps["out"].view(init_shape[0], -1)
        f_t_target_stack = self.target_encoder(x_t, fmaps=False).view(init_shape[0], -1)
        f_t = f_t_maps['f5']
        f_t = f_t.unsqueeze(1).view(init_shape[0], 4, *f_t.shape[1:])[:, -1]
        f_t = f_t.flatten(1, 2).transpose(-1, -2)
        f_t_stack = f_t_maps["out"].view(init_shape[0], 4, -1)
        f_t_global = f_t_stack[:, -1]

        if self.dense_supervision:
            all_states = all_states.view(init_shape[0]*n, *init_shape[2:])
            all_states = self.encoder(all_states)
            all_states = all_states.view(init_shape[0], n, -1)

        N = f_t_prev.size(0)
        reward_loss = 0
        local_sd_loss = 0

        if self.online_agent_training:
            self.agent.reset_noise()
            dqn_loss = self.agent.update(f_t_prev,
                                         actions[:, 0],
                                         (rewards - 1).float().sum(axis=-1),
                                         f_t_stack,
                                         done,
                                         1,
                                         step=False,
                                         n=n,
                                         target_next_states=f_t_target_stack,)
            trackers["online_dqn_loss"] += dqn_loss.mean().item()
        else:
            dqn_loss = 0

        # Do autoregressive jumps
        f_t_current = f_t_prev[:, -self.encoder.hidden_size:]
        current_stack = f_t_prev
        for i in range(actions.shape[1]):
            trackers["jumps"] += 1
            a_i = actions[:, i]
            r_i = rewards[:, i]
            reward_preds = self.reward_module(current_stack, a_i)
            if rewards.max() == 2:
                current_reward_loss = F.cross_entropy(reward_preds,
                                                      r_i,
                                                      weight=self.class_weights,
                                                      reduction="none")
            else:
                # If the batch contains no pos. reward, normalize manually
                current_reward_loss = F.cross_entropy(reward_preds,
                                                      r_i,
                                                      weight=self.class_weights,
                                                      reduction='none')
                current_reward_loss = current_reward_loss.sum() / (self.class_weights[r_i].sum() +
                                                            self.class_weights[2])
            reward_loss = reward_loss + current_reward_loss
            reward_preds = reward_preds.argmax(dim=-1)
            self.update_reward_trackers(i, r_i.detach(), reward_preds.detach(), mode)

            f_t_current = self.prediction_module(current_stack, a_i) + f_t_current
            if self.bilinear_global_loss:
                f_t_current = self.global_classifier(f_t_current)
            current_stack = torch.cat([current_stack[:, self.encoder.hidden_size:],
                                       f_t_current], -1)

            if self.dense_supervision or i == actions.shape[1] - 1:
                if i == actions.shape[1] - 1:
                    target = f_t_global
                else:
                    target = all_states[:, i]
                if self.detach_target:
                    sd_loss_target = target.detach()
                else:
                    sd_loss_target = target
                step_sd_loss = F.mse_loss(sd_loss_target, f_t_current,
                                          reduction="none").mean(-1)
                local_sd_loss = local_sd_loss + step_sd_loss

                pred_delta = f_t_current - f_t_initial
                true_delta = target - f_t_initial
                cos_sim = F.cosine_similarity(pred_delta, true_delta,
                                              dim=-1).mean()

                self.update_cos_sim_trackers(i, cos_sim, step_sd_loss.mean(), mode)

        if self.local_loss:
            f5_stack = f_t_prev_maps['f5'].unsqueeze(1).reshape(init_shape[0], 4, *f_t_prev_maps['f5'].shape[1:]).permute(0, 1, 4, 2, 3)
            f5_current = f5_stack[:, -1]
            f5_stack = f5_stack.flatten(1, 2)
            for i in range(actions.shape[1]):
                a_i = actions[:, i]
                f5_current = self.local_prediction_module(f5_stack, a_i) + f5_current

            f5_current = f5_current.flatten(2, 3).permute(0, 2, 1)
            predictions = self.local_classifier(f5_current).permute(0, 2, 1)
            local_local_loss = self.nce_per_location(predictions, f_t).mean(0)
            trackers["epoch_local_local_loss"] += local_local_loss.mean().detach().item()
        else:
            local_local_loss = 0

        f_t_pred = f_t_current
        # Loss 1: Global at time t, f5 patches at time t-1
        done_preds = self.done_module(current_stack, a_i)
        done_loss = F.cross_entropy(done_preds, done.long()[:, 0], reduction="none")
        done_preds = done_preds.argmax(dim=-1)
        self.update_done_trackers(done, done_preds, mode=mode)
        predictions = self.classifier(f_t_pred)
        loss1 = self.nce_with_negs_from_same_loc(predictions, f_t).mean(-1)

        if self.global_loss or self.bilinear_global_loss:
            logits = torch.matmul(f_t_pred, f_t_global.t())
            loss2 = F.cross_entropy(logits.t(),
                                    torch.arange(N).to(self.device),
                                    reduction="none")
            trackers["epoch_global_loss"] += loss2.mean().detach().item()
        else:
            loss2 = 0

        trackers["sd_loss"] += local_sd_loss.mean().item()
        trackers["true_representation_norm"] += torch.norm(f_t_global, dim=-1).mean().item()
        trackers["pred_representation_norm"] += torch.norm(f_t_pred[:f_t_global.shape[0]], dim=-1).mean().item()

        self.optimizer.zero_grad()
        base_loss = (loss1 +
                     loss2 +
                     local_local_loss +
                     reward_loss*self.reward_loss_weight +
                     done_loss.mean()*self.reward_loss_weight)
        if self.noncontrastive_global_loss:
            base_loss = base_loss + local_sd_loss*self.noncontrastive_loss_weight
        if self.online_agent_training:
            loss = (base_loss + dqn_loss *
                    self.dqn_loss_weight)/self.dqn_loss_weight
            self.agent.optimiser.zero_grad()
        else:
            loss = base_loss
        if mode == "train":
            (weights*loss).mean().backward()
            self.optimizer.step()
            if self.online_agent_training:
                self.agent.optimiser.step()

        trackers["epoch_done_loss"] += done_loss.mean().detach().item()
        trackers["epoch_loss"] += loss.mean().detach().item()
        trackers["epoch_local_loss"] += loss1.mean().detach().item()
        trackers["epoch_rew_loss"] += reward_loss.mean().detach().item()

        self.steps += 1
        if self.online_agent_training:
            self.maybe_update_target_net()

        # Don't take DQN into account for ES.
        return base_loss

    def train(self, tr_eps, val_eps=None, epochs=None):
        self.class_weights = self.generate_reward_class_weights(tr_eps,
                                                                self.no_class_weighting)
        self.update_target_net()
        if not epochs:
            epochs = self.epochs
        epochs = range(epochs)
        for _ in epochs:
            self.encoder.train(), self.classifier.train()
            self.do_one_epoch(tr_eps, log=True)

            if val_eps:
                self.encoder.eval(), self.classifier.eval()
                with torch.no_grad():
                    self.do_one_epoch(val_eps, log=True)

                if self.early_stopper.early_stop:
                    break

            self.epochs_till_now += 1

        # If we were doing early stopping, now is the time to load in the best params.
        if val_eps is not None:
            self.early_stopper.load_checkpoint(self)

        torch.save(self.encoder.state_dict(),
                   os.path.join(self.wandb.run.dir,
                                self.config['game'] + '.pt'))

    def predict(self, z, a, deterministic_rew=False, mean_rew=False):
        N = z.size(0)
        z_last = z.view(N, 4, -1)[:, -1, :]  # choose the last latent vector from z
        z = z.view(N, -1)
        new_states = self.prediction_module(z, a) + z_last
        if self.bilinear_global_loss:
            new_states = self.global_classifier(new_states)
        reward_predictions = self.reward_module(z, a)
        nonterminal_predictions = self.done_module(z, a)
        if deterministic_rew:
            rewards = reward_predictions.argmax() - 1
            nonterminals = nonterminal_predictions.argmax().float()
        elif mean_rew:
            weights = torch.arange(reward_predictions.shape[-1], device=z.device).float() - 1
            reward_predictions = torch.softmax(reward_predictions, -1)
            rewards = reward_predictions @ weights
            weights = torch.arange(nonterminal_predictions.shape[-1], device=z.device).float()
            nonterminal_predictions = torch.softmax(nonterminal_predictions, -1)
            nonterminals = nonterminal_predictions @ weights
        else:
            rewards = Categorical(logits=reward_predictions).sample() - 1
            nonterminals = Categorical(logits=nonterminal_predictions).sample()

        return new_states, rewards, nonterminals

    def log_results(self,
                    prefix="",
                    plots=False,
                    verbose_print=True,):

        if prefix == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers
        iterations = trackers["iterations"]
        if iterations == 0:
            # We did nothing since the last log, so just quit.
            self.reset_trackers(prefix)
            return

        rew_accs, pos_recalls, pos_precs, zero_recalls, zero_precs, \
        rew_acc, pos_recall, pos_precision, zero_recall, zero_precision, \
        cosine_sims, cosine_sim, sd_losses, sd_loss,\
        done_acc, done_recall, done_prec = self.summarize_trackers(prefix)
        local_loss = trackers["epoch_local_loss"] / iterations
        local_local_loss = trackers["epoch_local_local_loss"] / iterations
        reward_loss = trackers["epoch_rew_loss"] / iterations
        done_loss = trackers["epoch_done_loss"] / iterations
        global_loss = trackers["epoch_global_loss"] / iterations
        epoch_loss = trackers["epoch_loss"] / iterations
        online_dqn_loss = trackers["online_dqn_loss"] / iterations
        true_norm = trackers["true_representation_norm"] / iterations
        pred_norm = trackers["pred_representation_norm"] / iterations
        self.reset_trackers(prefix)
        print(
            "{} Epoch: {}, Epoch Loss: {:.3f}, Global-Local Loss: {:.3f}, Local-Local Loss: {:.3f}, Rew. Loss: {:.3f}, Done Loss: {:.3f}, Global Loss: {:.3f}, Dynamics Error: {:.3f}, Pred. cos. sim: {:.3f}, Rew. acc: {:.3f}, Done acc.: {:.3f}, DQN Loss: {:.3f} {}".format(
                prefix.capitalize(),
                self.epochs_till_now,
                epoch_loss,
                local_loss,
                local_local_loss,
                reward_loss,
                done_loss,
                global_loss,
                sd_loss,
                cosine_sim,
                rew_acc,
                done_acc,
                online_dqn_loss,
                prefix.capitalize()))
        print(
            "Pos. Rew. Recall: {:.3f}, Pos. Rew. Prec.: {:.3f}, Zero Rew. Recall: {:.3f}, Zero Rew. Prec.: {:.3f}, Done Recall: {:.3f}, Done Prec.: {:.3f}, Pred. Norm: {:.3f}, True Norm: {:.3f}".format(
                pos_recall,
                pos_precision,
                zero_recall,
                zero_precision,
                done_recall,
                done_prec,
                pred_norm,
                true_norm))

        for i in range(self.maximum_length - 1):
            jump = i + 1
            if verbose_print:
                print("At {} jumps: Pos. Recall: {:.3f}, Pos. Prec.: {:.3f}, Zero Recall: {:.3f}, Zero Prec.: {:.3f}, Rew. Acc.: {:.3f}, Cosine sim: {:.3f}, L2 error: {:.3f}".format(
                    jump,
                    pos_recalls[i],
                    pos_precs[i],
                    zero_recalls[i],
                    zero_precs[i],
                    rew_accs[i],
                    cosine_sims[i],
                    sd_losses[i]))

            self.wandb.log({prefix + 'Jump {} SD Cosine Similarity'.format(jump): cosine_sims[i],
                            prefix + 'Jump {} SD Loss'.format(jump): sd_losses[i],
                            prefix + "Jump {} Reward Accuracy".format(jump): rew_accs[i],
                            prefix + "Jump {} Pos. Reward Recall".format(jump): pos_recalls[i],
                            prefix + "Jump {} Zero Reward Recall".format(jump): zero_recalls[i],
                            prefix + "Jump {} Pos. Reward Precision".format(jump): pos_precs[i],
                            prefix + "Jump {} Zero Reward Precision".format(jump): zero_precs[i],
                            'FM epoch': self.epochs_till_now})

        self.wandb.log({prefix + ' loss': epoch_loss,
                        prefix + ' local loss': local_loss,
                        prefix + ' local-local loss': local_local_loss,
                        prefix + " Reward Loss": reward_loss,
                        prefix + " Done Loss": done_loss,
                        prefix + ' global loss': global_loss,
                        prefix + ' online DQN loss': online_dqn_loss,
                        prefix + " Reward Accuracy": rew_acc,
                        prefix + " Done Accuracy": done_acc,
                        prefix + ' SD Loss': sd_loss,
                        prefix + ' SD Cosine Similarity': cosine_sim,
                        prefix + " Pos. Reward Recall": pos_recall,
                        prefix + " Zero Reward Recall": zero_recall,
                        prefix + " Pos. Reward Precision": pos_precision,
                        prefix + " Zero Reward Precision": zero_precision,
                        prefix + " Done Recall": done_recall,
                        prefix + " Done Precision": done_prec,
                        prefix + " Pred norm": pred_norm,
                        prefix + " True norm": true_norm,
                        'FM epoch': self.epochs_till_now})

        if plots:
            dir = "./figs/{}/".format(self.wandb.run.name)
            try:
                os.makedirs(dir)
            except FileExistsError:
                # directory already exists
                pass
            images = []
            plt.figure()
            plt.plot(np.arange(len(rew_accs)), zero_recalls)
            plt.xlabel("Number of jumps")
            plt.ylabel("zero recall")
            plt.tight_layout()
            plt.savefig(dir+"{}_zero_recall_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} zero recall {}".format(prefix, self.epochs_till_now)))
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(rew_accs)), pos_recalls)
            plt.xlabel("Number of jumps")
            plt.ylabel("pos recall")
            plt.tight_layout()
            plt.savefig(dir+"{}_pos_recall_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} pos recall {}".format(prefix, self.epochs_till_now)))
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(rew_accs)), rew_accs)
            plt.xlabel("Number of jumps")
            plt.ylabel("Reward Accuracy")
            plt.tight_layout()
            plt.savefig(dir+"{}_rew_acc_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} rew acc {}".format(prefix, self.epochs_till_now)))
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(rew_accs)), pos_precs)
            plt.xlabel("Number of jumps")
            plt.ylabel("pos precision")
            plt.tight_layout()
            plt.savefig(dir+"{}_pos_prec_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} pos prec {}".format(prefix, self.epochs_till_now)))
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(rew_accs)), zero_precs)
            plt.xlabel("Number of jumps")
            plt.ylabel("zero precision")
            plt.tight_layout()
            plt.savefig(dir+"{}_zero_prec_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} zero prec {}".format(prefix, self.epochs_till_now)))
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(rew_accs)), cosine_sims)
            plt.xlabel("Number of jumps")
            plt.ylabel("cosine similarity")
            plt.tight_layout()
            plt.savefig(dir+"{}_cos_sim_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} cos sim {}".format(prefix, self.epochs_till_now)))
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(rew_accs)), sd_losses)
            plt.xlabel("Number of jumps")
            plt.ylabel("sd loss")
            plt.tight_layout()
            plt.savefig(dir+"{}_sd_loss_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} sd loss {}".format(prefix, self.epochs_till_now)))
            plt.close()

            labels = [
              "{} pos recall {}".format(prefix, self.epochs_till_now),
              "{} zero recall {}".format(prefix, self.epochs_till_now),
              "{} rew acc {}".format(prefix, self.epochs_till_now),
              "{} pos prec {}".format(prefix, self.epochs_till_now),
              "{} zero prec {}".format(prefix, self.epochs_till_now),
              "{} cos sim {}".format(prefix, self.epochs_till_now),
              "{} sd loss {}".format(prefix, self.epochs_till_now),
            ]
            log = {label: image for label, image in zip(labels, images)}
            log["FM epoch"] = self.epochs_till_now

            self.wandb.log(log)

