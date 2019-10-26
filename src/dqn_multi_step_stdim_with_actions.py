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
        self.layernorm = nn.LayerNorm(input_dim, elementwise_affine=False) \
            if layernorm else nn.Identity()
        self.conditioning = nn.Linear(cond_dim, input_dim*2)

    def forward(self, input, cond):
        conditioning = self.conditioning(cond)
        gamma = conditioning[..., :self.input_dim]
        beta = conditioning[..., self.input_dim:]

        return self.layernorm(input)*gamma + beta


class FILMPredictionModule(nn.Module):
    def __init__(self, state_dim, num_actions, layernorm=False):
        super().__init__()
        self.convert_actions = lambda a: F.one_hot(a, num_classes=num_actions)
        self.films = nn.ModuleList()
        self.films.append(FILM(state_dim*4, num_actions, layernorm=layernorm))
        self.films.append(FILM(state_dim*4, num_actions, layernorm=layernorm))
        self.films.append(FILM(state_dim*4, num_actions, layernorm=layernorm))
        self.network = nn.Sequential(
            nn.Linear(state_dim*4, state_dim*4),
            nn.ReLU(),
            nn.Linear(state_dim*4, state_dim*4),
            nn.ReLU(),
            nn.Linear(state_dim*4, state_dim))

    def forward(self, states, actions):
        actions = self.convert_actions(actions).float()
        current = self.films[0](states, actions)
        current = self.network[:2](current)
        current = self.films[1](current, actions)
        current = self.network[2:4](current)
        current = self.films[2](current, actions)
        return self.network[4:](current)


class FILMRewardPredictionModule(nn.Module):
    def __init__(self, state_dim, num_actions, reward_dim=3, layernorm=False):
        super().__init__()
        self.convert_actions = lambda a: F.one_hot(a, num_classes=num_actions)
        self.films = nn.ModuleList()
        self.films.append(FILM(state_dim*4, num_actions, layernorm=layernorm))
        self.films.append(FILM(state_dim*4, num_actions, layernorm=layernorm))
        self.films.append(FILM(state_dim*4, num_actions, layernorm=layernorm))
        self.network = nn.Sequential(
            nn.Linear(state_dim*4, state_dim*4),
            nn.ReLU(),
            nn.Linear(state_dim*4, state_dim*4),
            nn.ReLU(),
            nn.Linear(state_dim*4, reward_dim))

    def forward(self, states, actions):
        actions = self.convert_actions(actions).float()
        current = self.films[0](states, actions)
        current = self.network[:2](current)
        current = self.films[1](current, actions)
        current = self.network[2:4](current)
        current = self.films[2](current, actions)
        return self.network[4:](current)


class PredictionModule(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.convert_actions = lambda a: F.one_hot(a, num_classes=num_actions)
        self.network = nn.Sequential(
            nn.Linear(state_dim*num_actions*4, state_dim*4),
            nn.ReLU(),
            nn.Linear(state_dim*4, state_dim*4),
            nn.ReLU(),
            nn.Linear(state_dim*4, state_dim))

    def forward(self, states, actions):
        actions = self.convert_actions(actions).float()
        N = states.size(0)
        output = self.network(
            torch.bmm(states.unsqueeze(2), actions.unsqueeze(1)).view(N, -1))  # outer-product / bilinear integration, then flatten
        return output


class RewardPredictionModule(nn.Module):
    def __init__(self, state_dim, num_actions, reward_dim=3, dropout=0):
        super().__init__()
        self.convert_actions = lambda a: F.one_hot(a, num_classes=num_actions)
        self.network = nn.Sequential(
            nn.Linear(state_dim*num_actions*4, state_dim*4),
            nn.ReLU(),
            nn.Linear(state_dim*4, state_dim*4),
            nn.ReLU(),
            nn.Linear(state_dim*4, reward_dim))

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
        self.bilinear_global_loss = config['bilinear_global_loss']
        self.noncontrastive_global_loss = config['noncontrastive_global_loss']
        self.noncontrastive_loss_weight = config['noncontrastive_loss_weight']

        self.agent = agent
        self.online_agent_training = config["online_agent_training"]

        self.device = device
        self.classifier = nn.Linear(self.encoder.hidden_size, 64).to(device)
        self.global_classifier = nn.Linear(self.encoder.hidden_size,
                                           self.encoder.hidden_size).to(device)
        self.params = list(self.encoder.parameters())
        self.params += list(self.classifier.parameters())
        self.params += list(self.global_classifier.parameters())

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_update = self.config["target_update"]
        self.steps = 0

        if config["film"]:
            self.prediction_module = FILMPredictionModule(self.encoder.hidden_size,
                                                          config["num_actions"],
                                                          layernorm=config["layernorm"])
            self.reward_module = FILMRewardPredictionModule(self.encoder.hidden_size,
                                                            config["num_actions"],
                                                            layernorm=config["layernorm"])

        else:
            self.prediction_module = PredictionModule(self.encoder.hidden_size,
                                                      config["num_actions"])
            self.reward_module = RewardPredictionModule(self.encoder.hidden_size,
                                                        config["num_actions"])

        self.reward_loss_weight = config["reward_loss_weight"]

        self.dense_supervision = config["dense_supervision"]

        self.dqn_loss_weight = config["dqn_loss_weight"]

        self.prediction_module.to(device)
        self.reward_module.to(device)
        self.params += list(self.prediction_module.parameters())
        self.params += list(self.reward_module.parameters())
        self.hard_neg_factor = config["hard_neg_factor"]

        self.maximum_length = config["max_jump_length"]
        self.minimum_length = config["min_jump_length"]

        self.detach_target = config["detach_target"]

        self.optimizer = torch.optim.Adam(self.params, lr=config['encoder_lr'], eps=1e-5)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")
        self.epochs_till_now = 0

    def maybe_update_target_net(self):
        if self.steps > self.target_update:
            self.update_target_net()

    def update_target_net(self):
        self.steps = 0
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    def generate_batch(self, transitions):
        total_steps = len(transitions)
        print('Total Steps: {}'.format(len(transitions)))
        for idx in range(total_steps // self.batch_size):
            indices = np.random.randint(0, total_steps, size=self.batch_size)
            if self.minimum_length == self.maximum_length + 1:
                gap = self.maximum_length
            else:
                gap = np.random.randint(self.minimum_length,
                                        self.maximum_length + 1)
            t1 = indices - gap
            # don't allow negative indices.
            underflow = np.clip(t1, a_max=0, a_min=None)
            indices -= underflow
            t1 -= underflow
            x_t, x_tnext, a_t, r_tnext, dones = [], [], [], [], []
            all_states = []
            for t1, t2 in zip(t1, indices):
                # Get one sample from this episode
                # If our sample would cross an episode boundary, resample.
                while transitions[t2].timestep - gap < 0:
                    t2 = np.random.randint(0, total_steps)
                    t1 = t2 - gap
                    # don't allow negative indices.
                    underflow = np.clip(t1, a_max=0, a_min=None)
                    t1 -= underflow
                    t2 -= underflow

                trans = np.array([None] * 4)
                trans[-1] = transitions[t1]
                for i in range(4 - 2, -1, -1):  # e.g. 2 1 0
                    if trans[i + 1].timestep == 0:
                        trans[i] = blank_trans  # If future frame has timestep 0
                    else:
                        trans[i] = transitions[t1 - 4 + 1 + i]
                states = [t.state for t in trans]

                next_trans = np.array([None] * 4)
                next_trans[-1] = transitions[t2]
                for i in range(4 - 2, -1, -1):  # e.g. 2 1 0
                    if next_trans[i + 1].timestep == 0:
                        next_trans[i] = blank_trans  # If future frame has timestep 0
                    else:
                        next_trans[i] = transitions[t2 - 4 + 1 + i]
                next_states = [t.state for t in next_trans]

                actions = [t.action for t in transitions[t1:t2]]
                rewards = [t.reward + 1 for t in transitions[t1:t2]]

                x_t.append(torch.stack(states, 0))
                x_tnext.append(torch.stack(next_states, 0))
                # x_tnext.append(transitions[t2].state)
                if self.dense_supervision:
                    all_states.append(torch.stack([t.state for t in transitions[t1:t2]], 0))
                a_t.append(actions)
                r_tnext.append(rewards)
                dones.append(transitions[t2].nonterminal)

            if self.dense_supervision:
                all_states = torch.stack(all_states).to(self.device).float()/255.
            else:
                all_states = None

            yield torch.stack(x_t).to(self.device).float()/255.,\
                  torch.stack(x_tnext).to(self.device).float()/255.,\
                  torch.tensor(a_t, device=self.device).long(),\
                  torch.tensor(r_tnext, device=self.device).long(),\
                  torch.tensor(dones, device=self.device).unsqueeze(-1).float(),\
                  all_states,\
                  gap

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

    def generate_reward_class_weights(self, transitions):
        counts = [0, 0, 0]  # counts for reward=-1,0,1
        for trans in transitions:
            counts[trans.reward + 1] += 1

        weights = [0., 0., 0.]
        for i in range(3):
            if counts[i] != 0:
                weights[i] = sum(counts) / counts[i]
        return torch.tensor(weights, device=self.device)

    def update_reward_trackers(self, n, rewards, reward_preds):
        self.pos_rew_tps[n] += ((reward_preds == 2)*(rewards == 2)).float().sum()
        self.pos_rew_fps[n] += ((reward_preds == 2)*(rewards != 2)).float().sum()
        self.pos_rew_fns[n] += ((reward_preds != 2)*(rewards == 2)).float().sum()
        self.pos_rew_tns[n] += ((reward_preds != 2)*(rewards != 2)).float().sum()

        self.zero_rew_tps[n] += ((reward_preds == 1)*(rewards == 1)).float().sum()
        self.zero_rew_fps[n] += ((reward_preds == 1)*(rewards != 1)).float().sum()
        self.zero_rew_fns[n] += ((reward_preds != 1)*(rewards == 1)).float().sum()
        self.zero_rew_tns[n] += ((reward_preds != 1)*(rewards != 1)).float().sum()

    def update_cos_sim_trackers(self, n, cos_sim, sd_loss):
        self.cos_sims[n] += cos_sim
        self.sd_losses[n] += sd_loss
        self.counts[n] += 1

    def summarize_trackers(self):
        pos_recalls = []
        pos_precs = []
        zero_recalls = []
        zero_precs = []
        rew_accs = []
        for i in range(self.maximum_length):
            pos_recalls.append(self.pos_rew_tps[i] / (self.pos_rew_fns[i] + self.pos_rew_tps[i]))
            pos_precs.append(self.pos_rew_tps[i] / (self.pos_rew_tps[i] + self.pos_rew_fps[i]))

            zero_recalls.append(self.zero_rew_tps[i] / (self.zero_rew_fns[i] + self.zero_rew_tps[i]))
            zero_precs.append(self.zero_rew_tps[i] / (self.zero_rew_tps[i] + self.zero_rew_fps[i]))
            acc = (self.pos_rew_tps[i] + self.pos_rew_tns[i]) / \
                  (self.pos_rew_fns[i] + self.pos_rew_tps[i] +
                   self.pos_rew_fps[i] + self.pos_rew_tns[i])
            rew_accs.append(acc)

        pos_recall = (np.sum(self.pos_rew_tps) / (np.sum(self.pos_rew_fns) + np.sum(self.pos_rew_tps)))
        pos_prec = (np.sum(self.pos_rew_tps) / (np.sum(self.pos_rew_tps) + np.sum(self.pos_rew_fps)))
        zero_recall = (np.sum(self.zero_rew_tps) / (np.sum(self.zero_rew_fns) + np.sum(self.zero_rew_tps)))
        zero_prec = (np.sum(self.zero_rew_tps) / (np.sum(self.zero_rew_tps) + np.sum(self.zero_rew_fps)))
        acc = (np.sum(self.pos_rew_tps) + np.sum(self.pos_rew_tns)) / \
              (np.sum(self.pos_rew_fns) + np.sum(self.pos_rew_tps) +
               np.sum(self.pos_rew_fps) + np.sum(self.pos_rew_tns))

        cosine_sim = np.sum(self.cos_sims)/np.sum(self.counts)
        cosine_sims = self.cos_sims/self.counts
        sd_losses = self.sd_losses/self.counts
        sd_loss = np.sum(self.sd_losses/np.sum(self.counts))

        return rew_accs, pos_recalls, pos_precs, zero_recalls, zero_precs,\
               acc, pos_recall, pos_prec, zero_recall, zero_prec, \
               cosine_sims, cosine_sim, sd_losses, sd_loss

    def reset_trackers(self,):
        self.cos_sims = np.zeros(self.maximum_length)
        self.counts = np.zeros(self.maximum_length)
        self.pos_rew_tps = np.zeros(self.maximum_length)
        self.zero_rew_tps = np.zeros(self.maximum_length)
        self.pos_rew_fps = np.zeros(self.maximum_length)
        self.zero_rew_fps = np.zeros(self.maximum_length)
        self.pos_rew_fns = np.zeros(self.maximum_length)
        self.zero_rew_fns = np.zeros(self.maximum_length)
        self.pos_rew_tns = np.zeros(self.maximum_length)
        self.zero_rew_tns = np.zeros(self.maximum_length)
        self.sd_losses = np.zeros(self.maximum_length)

    def do_one_epoch(self, episodes, plots=False):
        mode = "train" if self.encoder.training else "val"
        epoch_loss, steps = 0., 0.
        epoch_local_loss, epoch_rew_loss, epoch_global_loss, rew_acc, = 0., 0., 0., 0.
        sd_loss = 0
        true_representation_norm = 0
        pred_representation_norm = 0
        online_dqn_loss = 0
        jumps = 0
        self.reset_trackers()

        data_generator = self.generate_batch(episodes)
        for x_tprev, x_t, actions, rewards, done, all_states, n in data_generator:
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
                                             target_next_states=f_t_target_stack,).mean()
                online_dqn_loss += dqn_loss
            else:
                dqn_loss = 0

            # Do autoregressive jumps
            f_t_current = f_t_prev[:, -self.encoder.hidden_size:]
            current_stack = f_t_prev
            for i in range(actions.shape[1]):
                jumps += 1
                a_i = actions[:, i]
                r_i = rewards[:, i]
                reward_preds = self.reward_module(current_stack, a_i)
                if rewards.max() == 2:
                    reward_loss = F.cross_entropy(reward_preds,
                                                  r_i,
                                                  weight=self.class_weights)
                else:
                    # If the batch contains no pos. reward, normalize manually
                    current_reward_loss = F.cross_entropy(reward_preds,
                                                          r_i,
                                                          weight=self.class_weights,
                                                          reduction='none')
                    reward_loss += current_reward_loss.sum() / (self.class_weights[r_i].sum() +
                                                                self.class_weights[2])
                reward_preds = reward_preds.argmax(dim=-1)
                self.update_reward_trackers(i, r_i, reward_preds)

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
                                       reduction="mean")
                    local_sd_loss = local_sd_loss + step_sd_loss

                    pred_delta = f_t_current - f_t_initial
                    true_delta = target - f_t_initial
                    cos_sim = F.cosine_similarity(pred_delta, true_delta,
                                                  dim=-1).mean()

                    self.update_cos_sim_trackers(i, cos_sim, step_sd_loss)

            f_t_pred = f_t_current
            # Loss 1: Global at time t, f5 patches at time t-1
            predictions = self.classifier(f_t_pred)
            f_t = f_t.flatten(1, 2).transpose(-1, -2)
            loss1 = self.nce_with_negs_from_same_loc(predictions, f_t).mean()

            if self.global_loss or self.bilinear_global_loss:
                logits = torch.matmul(f_t_pred, f_t_global.t())
                loss2 = F.cross_entropy(logits.t(),
                                        torch.arange(N).to(self.device))
                epoch_global_loss += loss2.detach().item()
            else:
                loss2 = 0

            sd_loss += local_sd_loss
            true_representation_norm += torch.norm(f_t_global, dim=-1).mean()
            pred_representation_norm += torch.norm(f_t_pred[:f_t_global.shape[0]], dim=-1).mean()

            self.optimizer.zero_grad()
            loss = (loss1 +
                    loss2 +
                    reward_loss*self.reward_loss_weight)
            if self.noncontrastive_global_loss:
                loss = loss + local_sd_loss*self.noncontrastive_loss_weight
            if self.online_agent_training:
                loss = (loss + dqn_loss *
                        self.dqn_loss_weight)/self.dqn_loss_weight
                self.agent.optimiser.zero_grad()
            if mode == "train":
                loss.backward()
                self.optimizer.step()
                if self.online_agent_training:
                    self.agent.optimiser.step()

            epoch_loss += loss.detach().item()
            epoch_local_loss += loss1.detach().item()
            epoch_rew_loss += reward_loss.detach().item()

            steps += 1
            self.steps += 1
            self.maybe_update_target_net()

        rew_accs, pos_recalls, pos_precs, zero_recalls, zero_precs, \
        rew_acc, pos_recall, pos_prec, zero_recall, zero_prec, \
        cosine_sims, cosine_sim, sd_losses, sd_loss = self.summarize_trackers()

        self.log_results(epoch_local_loss / steps,
                         epoch_rew_loss / steps,
                         epoch_global_loss / steps,
                         epoch_loss / steps,
                         online_dqn_loss / steps,
                         sd_loss,
                         cosine_sim,
                         rew_acc,
                         pos_recall,
                         pos_prec,
                         zero_recall,
                         zero_prec,
                         true_representation_norm / steps,
                         pred_representation_norm / steps,
                         rew_accs,
                         pos_recalls,
                         pos_precs,
                         zero_recalls,
                         zero_precs,
                         cosine_sims,
                         sd_losses,
                         prefix=mode,
                         plots=plots)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    def train(self, tr_eps, val_eps=None, epochs=None):
        self.class_weights = self.generate_reward_class_weights(tr_eps)
        self.update_target_net()
        if not epochs:
            epochs = self.epochs
        epochs = range(epochs)
        for _ in epochs:
            self.encoder.train(), self.classifier.train()
            self.do_one_epoch(tr_eps)

            if val_eps:
                self.encoder.eval(), self.classifier.eval()
                with torch.no_grad():
                    self.do_one_epoch(val_eps)

                if self.early_stopper.early_stop:
                    break
            self.epochs_till_now += 1
        torch.save(self.encoder.state_dict(),
                   os.path.join(self.wandb.run.dir,
                                self.config['game'] + '.pt'))

    def predict(self, z, a, deterministic_rew=False):
        N = z.size(0)
        z_last = z.view(N, 4, -1)[:, -1, :]  # choose the last latent vector from z
        z = z.view(N, -1)
        new_states = self.prediction_module(z, a) + z_last
        if self.bilinear_global_loss:
            new_states = self.global_classifier(new_states)
        reward_predictions = self.reward_module(z, a)
        if deterministic_rew:
            rewards = reward_predictions.argmax() - 1
        else:
            rewards = Categorical(logits=reward_predictions).sample() - 1

        return new_states, rewards

    def log_results(self,
                    local_loss,
                    reward_loss,
                    global_loss,
                    epoch_loss,
                    online_dqn_loss,
                    sd_loss,
                    sd_cosine_sim,
                    rew_acc,
                    pos_recall,
                    pos_precision,
                    zero_recall,
                    zero_precision,
                    true_norm,
                    pred_norm,
                    rew_accs,
                    pos_recalls,
                    pos_precs,
                    zero_recalls,
                    zero_precs,
                    cosine_sims,
                    sd_losses,
                    prefix="",
                    verbose_print=True,
                    plots=False):
        print(
            "{} Epoch: {}, Epoch Loss: {:.3f}, Local Loss: {:.3f}, Reward Loss: {:.3f}, Global Loss: {:.3f}, Dynamics Error: {:.3f}, Prediction Cosine Similarity: {:.3f}, Reward Accuracy: {:.3f}, DQN Loss: {:.3f} {}".format(
                prefix.capitalize(),
                self.epochs_till_now,
                epoch_loss,
                local_loss,
                reward_loss,
                global_loss,
                sd_loss,
                sd_cosine_sim,
                rew_acc,
                online_dqn_loss,
                prefix.capitalize()))
        print(
            "Pos. Rew. Recall: {:.3f}, Pos. Rew. Prec.: {:.3f}, Zero Rew. Recall: {:.3f}, Zero Rew. Prec.: {:.3f}, Pred. Norm: {:.3f}, True Norm: {:.3f}".format(
                pos_recall,
                pos_precision,
                zero_recall,
                zero_precision,
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
                        prefix + " Reward Loss": reward_loss,
                        prefix + ' global loss': global_loss,
                        prefix + " Reward Accuracy": rew_acc,
                        prefix + ' SD Loss': sd_loss,
                        prefix + ' SD Cosine Similarity': sd_cosine_sim,
                        prefix + " Pos. Reward Recall": pos_recall,
                        prefix + " Zero Reward Recall": zero_recall,
                        prefix + " Pos. Reward Precision": pos_precision,
                        prefix + " Zero Reward Precision": zero_precision,
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
            fig = plt.figure()
            plt.plot(np.arange(len(rew_accs)), zero_recalls)
            plt.xlabel("Number of jumps")
            plt.ylabel("zero recall")
            plt.tight_layout()
            plt.savefig(dir+"{}_zero_recall_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} zero recall {}".format(prefix, self.epochs_till_now)))
            plt.clf()

            fig = plt.figure()
            plt.plot(np.arange(len(rew_accs)), pos_recalls)
            plt.xlabel("Number of jumps")
            plt.ylabel("pos recall")
            plt.tight_layout()
            plt.savefig(dir+"{}_pos_recall_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} pos recall {}".format(prefix, self.epochs_till_now)))
            plt.clf()

            fig = plt.figure()
            plt.plot(np.arange(len(rew_accs)), rew_accs)
            plt.xlabel("Number of jumps")
            plt.ylabel("Reward Accuracy")
            plt.tight_layout()
            plt.savefig(dir+"{}_rew_acc_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} rew acc {}".format(prefix, self.epochs_till_now)))
            plt.clf()

            fig = plt.figure()
            plt.plot(np.arange(len(rew_accs)), pos_precs)
            plt.xlabel("Number of jumps")
            plt.ylabel("pos precision")
            plt.tight_layout()
            plt.savefig(dir+"{}_pos_prec_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} pos prec {}".format(prefix, self.epochs_till_now)))
            plt.clf()

            fig = plt.figure()
            plt.plot(np.arange(len(rew_accs)), zero_precs)
            plt.xlabel("Number of jumps")
            plt.ylabel("zero precision")
            plt.tight_layout()
            plt.savefig(dir+"{}_zero_prec_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} zero prec {}".format(prefix, self.epochs_till_now)))
            plt.clf()

            fig = plt.figure()
            plt.plot(np.arange(len(rew_accs)), cosine_sims)
            plt.xlabel("Number of jumps")
            plt.ylabel("cosine similarity")
            plt.tight_layout()
            plt.savefig(dir+"{}_cos_sim_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} cos sim {}".format(prefix, self.epochs_till_now)))
            plt.clf()

            fig = plt.figure()
            plt.plot(np.arange(len(rew_accs)), sd_losses)
            plt.xlabel("Number of jumps")
            plt.ylabel("sd loss")
            plt.tight_layout()
            plt.savefig(dir+"{}_sd_loss_{}.png".format(prefix, self.epochs_till_now))
            image = save_to_pil()
            images.append(wandb.Image(image,
                                      caption="{} sd loss {}".format(prefix, self.epochs_till_now)))

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

