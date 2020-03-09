from torch import nn
from torch.distributions import Categorical
import torch
from torch.nn import functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.synchronize import find_port
from src.logging import update_trackers, reset_trackers
from src.mcts_memory import AsyncPrioritizedSequenceReplayFrameBufferExtended, initialize_replay_buffer
import numpy as np
import sys
import traceback
import gym
import copy
import time
import os
import dill

try:
    from apex import amp
except ModuleNotFoundError as e:
    print("Could not import AMP: mixed precision will fail.")


NetworkOutput = namedarraytuple('NetworkOutput', ['next_state', 'reward', 'policy_logits', 'value'])


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_small = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), gain=0.01)
init_0 = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0))
init_relu = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                           constant_(x, 0), nn.init.calculate_gain('relu'))


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size, seed, port, backend="nccl"):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend,
                            rank=rank,
                            world_size=world_size,
                            init_method=f"tcp://127.0.0.1:{port}",
                            )

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(seed)


def create_network(args):
    dummy_env = gym.vector.make('atari-v0', num_envs=1, args=args,
                                asynchronous=False)
    dummy_env.seed(args.seed)
    model = MCTSModel(args, dummy_env.action_space[0].n)
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay,
                                      eps=args.adam_eps)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay ** (1. / args.lr_decay_steps),)
    model.to(args.device)
    model.share_memory()
    return model, optimizer, None


class TrainingWorker(object):
    def __init__(self, rank, size, args, port, squeue, error_queue,
                     receive_queue,  backend="nccl",):
        super().__init__()
        self.args = args
        self.size = size
        self.rank = rank
        self.backend = backend
        self.port = port

        self.squeue = squeue
        self.error_queue = error_queue
        self.receive_queue = receive_queue

        self.epochs_till_now = 0

        self.maximum_length = args.jumps
        self.multistep = args.multistep
        self.use_all_targets = args.use_all_targets
        self.train_trackers = reset_trackers(self.maximum_length)

    def startup(self):
        if torch.cuda.is_available():
            self.devices = torch.cuda.device_count()
            if not self.args.no_gpu_0_train:
                device_id = self.rank % self.devices
            else:
                device_id = (self.rank % (self.devices - 1))+1
            self.args.device = torch.device('cuda:{}'.format(device_id))
            torch.cuda.set_device(self.args.device)
            torch.cuda.manual_seed(self.args.seed)
            torch.backends.cudnn.enabled = True
        else:
            self.args.device = torch.device('cpu')
            self.backend = 'gloo'

        setup(self.rank, self.size, self.args.seed, self.port, self.backend)
        self.model, self.optimizer, self.scheduler = create_network(self.args)
        print("{} started on gpu {}".format(self.rank, self.args.device),flush=True)
        if self.rank == 0:
            print("{} sending model".format(self.rank),flush=True)
            self.squeue.put(self.model)

        if self.args.fp16:
            amp.initialize(self.model, self.optimizer)
        if self.args.num_trainers > 1:
            print("{} initializing DDP".format(self.rank), flush=True)
            self.model = DDP(self.model,
                             device_ids=[self.args.device],
                             output_device=self.args.device,
                             find_unused_parameters=True)

    def optimize(self, buffer):
        self.buffer = buffer.x
        print("{} starting up".format(self.rank), flush=True)
        self.startup()
        try:
            while True:
                env_steps = self.receive_queue.get()
                self.train(self.args.epoch_steps, log=self.rank == 0)

                if self.rank == 0:
                    self.squeue.put((self.epochs_till_now, self.train_trackers))
                    self.train_trackers = reset_trackers(self.maximum_length)
        except (KeyboardInterrupt, Exception):
            print(sys.exc_info(), flush=True)
            traceback.print_exc()
        finally:
            return

    def train(self, steps, log=True):
        for step in range(steps):
            self.epochs_till_now += 1
            loss = self.model(self.buffer,
                              self.train_trackers if log else None)

            loss = loss.mean()
            self.optimizer.zero_grad()
            if self.args.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            # self.scheduler.step()


class LocalNCE:
    def __init__(self, classifier, temperature=0.1):
        self.classifier = classifier
        self.inv_temp = 1/temperature

    def calculate_accuracy(self, preds):
        labels = torch.arange(preds.shape[1], dtype=torch.long, device=preds.device)
        preds = torch.argmax(-preds, dim=-1)
        acc = float(torch.sum(torch.eq(labels, preds)).data) / preds.numel()
        return acc

    def forward(self, f_x1, f_x2):
        """
        Compute InfoNCE cost with source features in f_x1 and target features in
        f_x2. We assume one source feature vector per location per item in batch
        and one target feature vector per location per item in batch. There are
        n_batch items, n_locs locations, and n_rkhs dimensions per vector.
        -- note: we can predict x1->x2 and x2->x1 in parallel "for free"

        For the positive nce pair (f_x1[i, :, l], f_x2[i, :, l]), which comes from
        batch item i at spatial location l, we will use the target feature vectors
        f_x2[j, :, l] as negative samples, for all j != i.

        Input:
          f_x1 : (n_locs, n_batch, n_rkhs)  -- n_locs source vectors per item
          f_x2 : (n_locs, n_batch, n_rkhs)  -- n_locs target vectors per item.  Negative samples.
        Output:
          loss_nce : (n_batch, n_locs)       -- InfoNCE cost at each location
        """
        # reshaping for big matrix multiply
        # f_x1 = f_x1.permute(2, 0, 1)  # (n_locs, n_batch, n_rkhs)
        f_x1 = self.classifier(f_x1)
        f_x2 = f_x2.permute(0, 2, 1)  # (n_locs, n_rkhs, n_batch)

        f_x1 = f_x1/(torch.norm(f_x1, dim=-1, keepdim=True) + 1.e-3)
        f_x2 = f_x2/(torch.norm(f_x2, dim=-1, keepdim=True) + 1.e-3)

        n_batch = f_x1.size(1)
        neg_batch = f_x2.size(-1)

        # compute dot(f_glb[i, :, l], f_lcl[j, :, l]) for all i, j, l
        # -- after matmul: raw_scores[l, i, j] = dot(f_x1[i, :, l], f_x2[j, :, l])
        raw_scores = torch.matmul(f_x1, f_x2)*self.inv_temp  # (n_locs, n_batch, n_batch)
        # We get NCE log softmax by normalizing over dim 1 or 2 of raw_scores...
        # -- normalizing over dim 1 gives scores for predicting x2->x1
        # -- normalizing over dim 2 gives scores for predicting x1->x2
        lsmax_x1_to_x2 = -F.log_softmax(raw_scores, dim=2)  # (n_locs, n_batch, n_batch)
        lsmax_x2_to_x1 = -F.log_softmax(raw_scores, dim=1)  # (n_locs, n_batch, n_batch)
        # make a mask for picking out the NCE scores for positive pairs
        pos_mask = torch.eye(n_batch, dtype=f_x1.dtype, device=f_x1.device)
        if n_batch != neg_batch:
            with torch.no_grad():
                mask = torch.zeros((n_batch, neg_batch), dtype=f_x1.dtype, device=f_x1.device)
                mask[:n_batch, :n_batch] += pos_mask
                pos_mask = mask
        pos_mask = pos_mask.unsqueeze(dim=0)
        # use masked sums to select NCE scores for positive pairs
        loss_nce_x1_to_x2 = (lsmax_x1_to_x2 * pos_mask).sum(dim=2)  # (n_locs, n_batch)
        loss_nce_x2_to_x1 = (lsmax_x2_to_x1 * pos_mask).sum(dim=1)[:, :n_batch]  # (n_locs, n_batch)
        # combine forwards/backwards prediction costs (or whatever)
        loss_nce = 0.5 * (loss_nce_x1_to_x2 + loss_nce_x2_to_x1)
        acc = self.calculate_accuracy(lsmax_x1_to_x2)
        return loss_nce, acc


class BlockNCE:
    def __init__(self, classifier, temperature=0.1, use_self_targets=False):
        self.classifier = classifier
        self.inv_temp = 1/temperature
        self.use_self_targets = use_self_targets

    def calculate_accuracy(self, preds):
        labels = torch.arange(preds.shape[-2], dtype=torch.long, device=preds.device)
        preds = torch.argmax(-preds, dim=-1)
        corrects = torch.eq(labels, preds)
        return corrects

    def forward(self, f_x1s, f_x2s):
        """
        Compute InfoNCE cost with source features in f_x1 and target features in
        f_x2. We assume one source feature vector per location per item in batch
        and one target feature vector per location per item in batch. There are
        n_batch items, n_locs locations, and n_rkhs dimensions per vector.
        -- note: we can predict x1->x2 and x2->x1 in parallel "for free"

        For the positive nce pair (f_x1[i, :, l], f_x2[i, :, l]), which comes from
        batch item i at spatial location l, we will use the target feature vectors
        f_x2[j, :, l] as negative samples, for all j != i.

        Input:
          f_x1 : (n_locs, n_times, n_batch, n_rkhs)  -- n_locs source vectors per item
          f_x2 : (n_locs, n_times, n_batch, n_rkhs)  -- n_locs target vectors per item.  Negative samples.
        Output:
          loss_nce : (n_batch, n_locs)       -- InfoNCE cost at each location
        """
        # reshaping for big matrix multiply
        # f_x1 = f_x1.permute(2, 0, 1)  # (n_locs, n_batch, n_rkhs)
        f_x1 = f_x1s.flatten(1, 2)
        f_x2 = f_x2s.flatten(1, 2)
        f_x1 = self.classifier(f_x1)
        f_x1 = f_x1/(torch.norm(f_x1, dim=-1, keepdim=True) + 1.e-3)
        f_x2 = f_x2/(torch.norm(f_x2, dim=-1, keepdim=True) + 1.e-3)
        f_x2 = f_x2.permute(0, 2, 1)  # (n_locs, n_rkhs, n_batch)
        n_batch = f_x1.size(1)
        neg_batch = f_x2.size(-1)

        # compute dot(f_glb[i, :, l], f_lcl[j, :, l]) for all i, j, l
        # -- after matmul: raw_scores[l, i, j] = dot(f_x1[i, :, l], f_x2[j, :, l])
        raw_scores = torch.matmul(f_x1, f_x2)*self.inv_temp  # (n_locs, n_batch, n_batch)

        # make a mask for picking out the NCE scores for positive pairs

        pos_mask = torch.eye(n_batch, dtype=f_x1.dtype, device=f_x1.device)
        if n_batch != neg_batch:
            with torch.no_grad():
                mask = torch.zeros((n_batch, neg_batch),
                                   dtype=f_x1.dtype,
                                   device=f_x1.device)
                mask[:n_batch, :n_batch] += pos_mask
                pos_mask = mask

        pos_mask = pos_mask.unsqueeze(dim=0)
        if not self.use_self_targets:
            t = f_x1s.shape[1]
            batch_mask = torch.eye(f_x1s.shape[2],
                                   dtype=f_x1.dtype,
                                   device=f_x1.device)
            batch_mask = batch_mask[None, :, None, :].expand(t, -1, t, -1)
            batch_mask = batch_mask.flatten(0, 1).flatten(1, 2)
            if n_batch != neg_batch:
                with torch.no_grad():
                    mask = torch.zeros((n_batch, neg_batch),
                                       dtype=f_x1.dtype,
                                       device=f_x1.device)
                    mask[:n_batch, :n_batch] += batch_mask
                    batch_mask = mask
            batch_mask = batch_mask.unsqueeze(dim=0)
            weight_mask = 1 - (batch_mask - pos_mask)
            raw_scores = raw_scores * weight_mask

        # We get NCE log softmax by normalizing over dim 1 or 2 of raw_scores...
        # -- normalizing over dim 1 gives scores for predicting x2->x1
        # -- normalizing over dim 2 gives scores for predicting x1->x2

        lsmax_x1_to_x2 = -F.log_softmax(raw_scores, dim=2)  # (n_locs, n_batch, n_batch)
        lsmax_x2_to_x1 = -F.log_softmax(raw_scores, dim=1)  # (n_locs, n_batch, n_batch)
        # use masked sums to select NCE scores for positive pairs
        loss_nce_x1_to_x2 = (lsmax_x1_to_x2 * pos_mask).sum(dim=2)  # (n_locs, n_batch)
        loss_nce_x2_to_x1 = (lsmax_x2_to_x1 * pos_mask).sum(dim=1)[:, :n_batch]  # (n_locs, n_batch)
        # combine forwards/backwards prediction costs (or whatever)
        loss_nce = 0.5 * (loss_nce_x1_to_x2 + loss_nce_x2_to_x1)
        loss_nce = loss_nce.view(*f_x1.shape[0:2])
        corrects = self.calculate_accuracy(lsmax_x1_to_x2)
        accuracy = torch.mean(corrects.float().view(*f_x1s.shape[:3]), (0, 2)).detach().cpu().numpy()
        return loss_nce.mean(0).view(f_x1s.shape[1], f_x1s.shape[2]), accuracy


class MCTSModel(nn.Module):
    def __init__(self, args, num_actions):
        super().__init__()
        self.args = args
        self.jumps = args.jumps
        self.multistep = args.multistep
        self.use_all_targets = args.use_all_targets
        self.no_nce = args.no_nce
        self.total_steps = 0

        self.batch_range = torch.arange(args.batch_size_per_worker).to(self.args.device)
        if args.film:
            self.dynamics_model = FiLMTransitionModel(channels=args.hidden_size,
                                                      cond_size=num_actions,
                                                      blocks=args.dynamics_blocks,
                                                      args=args)
        else:
            self.dynamics_model = TransitionModel(channels=args.hidden_size,
                                                  num_actions=num_actions,
                                                  blocks=args.dynamics_blocks,
                                                  args=args)
        if self.args.q_learning:
            self.value_model = QNetwork(args.hidden_size, num_actions)
        else:
            self.value_model = ValueNetwork(args.hidden_size)
            self.policy_model = PolicyNetwork(args.hidden_size, num_actions)
        self.encoder = RepNet(args.framestack, grayscale=args.grayscale, actions=False)
        if not self.no_nce:
            self.target_encoder = SmallEncoder(args)
            self.classifier = nn.Sequential(nn.Linear(args.hidden_size,
                                                      args.hidden_size),
                                            nn.ReLU())
            self.nce = BlockNCE(self.classifier, use_self_targets=args.use_all_targets)

        self.use_target_network = args.local_target_net
        if self.use_target_network:
            self.target_repnet = copy.deepcopy(self.encoder)
            self.target_value_model = copy.deepcopy(self.value_model)

    def update_target_network(self, steps):
        if steps % self.args.target_update_interval == 0 and self.use_target_network:
            self.target_repnet.load_state_dict(self.encoder.state_dict())
            self.target_value_model.load_state_dict(self.value_model.state_dict())

    def encode(self, images, actions):
        return self.encoder(images, actions)

    def initial_inference(self, obs, actions=None, logits=False):
        if len(obs.shape) < 5:
            obs = obs.unsqueeze(0)
        obs = obs.flatten(1, 2)
        hidden_state = self.encoder(obs, actions)
        if not self.args.q_learning:
            policy_logits = self.policy_model(hidden_state)
        else:
            policy_logits = None
        value_logits = self.value_model(hidden_state)
        reward_logits = self.dynamics_model.reward_predictor(hidden_state)

        if logits:
            return NetworkOutput(hidden_state, reward_logits, policy_logits, value_logits)

        value = inverse_transform(from_categorical(value_logits, logits=True))
        reward = inverse_transform(from_categorical(reward_logits, logits=True))
        return NetworkOutput(hidden_state, reward, policy_logits, value)

    def value_target_network(self, obs, actions):
        if len(obs.shape) < 5:
            obs = obs.unsqueeze(0)
        obs = obs.flatten(1, 2)
        hidden_state = self.target_repnet(obs, actions)
        value_logits = self.target_value_model(hidden_state)
        value = inverse_transform(from_categorical(value_logits,
                                                   logits=True))
        return value

    def inference(self, state, action):
        next_state, reward_logits, \
        policy_logits, value_logits = self.step(state, action)
        value = inverse_transform(from_categorical(value_logits,
                                                   logits=True))
        reward = inverse_transform(from_categorical(reward_logits,
                                                    logits=True))

        return NetworkOutput(next_state, reward, policy_logits, value)

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        if not self.args.q_learning:
            policy_logits = self.policy_model(next_state)
        else:
            policy_logits = None
        value_logits = self.value_model(next_state)

        return next_state, reward_logits, policy_logits, value_logits

    def forward(self, buffer, trackers=None, step=True,):
        """
        Do one update of the model on data drawn from a buffer.
        :param step: whether or not to actually take a gradient step.
        :return: Updated weights for prioritized experience replay.
        """
        with torch.no_grad():
            if self.args.prioritized:
                states, actions, rewards, return_, done, done_n, unk, \
                is_weights, policies, values = buffer.sample_batch(self.args.batch_size_per_worker)
                is_weights = is_weights.float().to(self.args.device)
            else:
                states, actions, rewards, return_, done, done_n, unk, \
                policies, values = buffer.sample_batch(self.args.batch_size_per_worker)
                is_weights = 1.

            target_images = states[0:self.jumps+1, :, 0].transpose(0, 1)
            target_images = target_images.reshape(-1, *states.shape[-3:])
            initial_states = states[0]
            target_images = target_images.to(self.args.device).float()/255.
            initial_states = initial_states.to(self.args.device).float() / 255.

            actions = actions.long().to(self.args.device)
            # Note that rewards are shifted right by one; r[i] is reward received when arriving at i.
            # Because of how this happens, we need to make sure that the first reward received isn't
            # actually from a different trajectory.  If it is, we just set it to 0.
            rewards[0] = rewards[0] * (1 - done[0].float())
            rewards = rewards.float().to(self.args.device)
            policies = policies.float().to(self.args.device)
            values = values.float().to(self.args.device)
            initial_actions = actions[0]

            if self.use_target_network:
                value_target_states = states[self.args.multistep:self.jumps+self.args.multistep+1]
                value_target_states = value_target_states.to(self.args.device).float()/255.
                value_targets = self.value_target_network(value_target_states.flatten(0, 1), None)
                value_targets = value_targets.view(*value_target_states.shape[0:2], -1)
                if self.args.q_learning:
                    value_targets = value_targets.max(dim=-1, keepdim=False)[0]
                values[self.args.multistep:self.jumps+self.args.multistep+1] = value_targets

        # Get into the shape used by the NCE code.
        if not self.no_nce:
            target_images = self.target_encoder(target_images)
            target_images = target_images.view(states.shape[1], -1, *target_images.shape[1:])
            target_images = target_images.flatten(3, 4).permute(3, 0, 1, 2)

        current_state, pred_reward,\
        pred_policy, pred_value = self.initial_inference(initial_states,
                                                         initial_actions,
                                                         logits=True)

        if self.args.q_learning:
            pred_policy = pred_value
            pred_value = pred_value[self.batch_range, actions[1], :]
            pred_policy = inverse_transform(from_categorical(pred_policy,
                                                             logits=True))

        # This represents s_1, r_0, pi_1, v_1
        predictions = [(1.0, pred_reward, pred_policy, pred_value)]
        pred_states = [current_state]

        pred_values, pred_rewards, pred_policies = [], [], []
        loss = torch.zeros(1, device=self.args.device)
        reward_losses, value_losses, policy_losses, \
            nce_losses, value_targets = [], [], [], [], []
        total_losses, nce_accs = np.zeros(self.jumps + 1),\
                                 np.zeros(self.jumps + 1)
        target_entropies, pred_entropies = [], []

        discounts = torch.ones_like(rewards)[:self.multistep]*self.args.discount
        discounts = discounts ** torch.arange(0, self.multistep, device=self.args.device)[:, None].float()

        value_errors, reward_errors = [], []

        for i in range(0, self.jumps):
            action = actions[i]
            current_state, pred_reward, pred_policy, pred_value = \
                self.step(current_state, action)
            if self.args.q_learning:
                pred_policy = pred_value
                pred_policy = inverse_transform(from_categorical(pred_policy,
                                                   logits=True))
                pred_value = pred_value[self.batch_range, actions[i+1], :]

            current_state = ScaleGradient.apply(current_state,
                                                self.args.grad_scale_factor)

            predictions.append((1. / self.jumps,
                                pred_reward,
                                pred_policy,
                                pred_value))
            pred_states.append(current_state)

        for i, prediction in enumerate(predictions):
            # recall that predictions_i is r_i, pi_i+1, v_i+1
            loss_scale, pred_reward, pred_policy, pred_value = prediction

            # Calculate the value target for v_i+1
            value_target = torch.sum(discounts*rewards[i+1:i+self.multistep+1], 0)
            value_target = value_target + self.args.discount ** self.multistep \
                           * values[i+self.multistep]

            value_targets.append(value_target)
            value_target = to_categorical(transform(value_target))
            reward_target = to_categorical(transform(rewards[i]))

            pred_rewards.append(inverse_transform(from_categorical(
                pred_reward.detach(), logits=True)))
            pred_values.append(inverse_transform(from_categorical(
                pred_value.detach(), logits=True)))

            pred_entropy = Categorical(logits=pred_policy).entropy()
            target_entropy = Categorical(probs=policies[i]).entropy()
            pred_entropies.append(pred_entropy.mean().cpu().detach().item())
            target_entropies.append(target_entropy.mean().cpu().detach().item())

            value_errors.append(torch.abs(pred_values[-1] - value_targets[-1]).detach().cpu().numpy())
            reward_errors.append(torch.abs(pred_rewards[-1] - rewards[i]).detach().cpu().numpy())

            pred_value = F.log_softmax(pred_value, -1)
            pred_reward = F.log_softmax(pred_reward, -1)
            pred_policy = F.log_softmax(pred_policy, -1)

            current_reward_loss = -torch.sum(reward_target * pred_reward, -1)
            current_value_loss = -torch.sum(value_target * pred_value, -1)
            current_policy_loss = -torch.sum(policies[i] * pred_policy, -1)

            loss = loss + loss_scale * (is_weights * (
                       current_value_loss*self.args.value_loss_weight +
                       current_policy_loss*self.args.policy_loss_weight +
                       current_reward_loss*self.args.reward_loss_weight -
                       pred_entropy * self.args.entropy_loss_weight).mean())

            total_losses[i] += (current_value_loss*self.args.value_loss_weight +
                                current_policy_loss*self.args.policy_loss_weight +
                                current_reward_loss*self.args.reward_loss_weight +
                                pred_entropy*self.args.entropy_loss_weight).detach().mean().cpu().item()

            reward_losses.append(current_reward_loss.detach().mean().cpu().item())
            value_losses.append(current_value_loss.detach().mean().cpu().item())
            policy_losses.append(current_policy_loss.detach().mean().cpu().item())

        if not self.no_nce:
            # if self.use_all_targets:
            target_images = target_images.permute(0, 2, 1, 3)
            nce_input = torch.stack(pred_states, 1).flatten(3, 4).permute(3, 1, 0, 2)
            nce_loss, nce_accs = self.nce.forward(nce_input, target_images)
            nce_losses = nce_loss.mean(-1).detach().cpu().numpy()
            nce_loss = (nce_loss*is_weights).mean(-1)
            for i, current_nce_loss in enumerate(nce_loss):
                loss_scale = predictions[i][0]
                loss = loss + loss_scale*self.args.contrastive_loss_weight*current_nce_loss
                total_losses[i] += current_nce_loss.mean().detach().cpu().item()

        else:
            nce_losses = np.zeros(self.jumps + 1)

        if self.args.prioritized:
            buffer.update_batch_priorities(value_errors[0] + 1e-5)

        mean_values = torch.mean(torch.stack(pred_values, 0), -1).detach().cpu().numpy()
        mean_rewards = torch.mean(torch.stack(pred_rewards, 0), -1).detach().cpu().numpy()
        target_values = torch.mean(torch.stack(value_targets, 0), -1).detach().cpu().numpy()
        target_rewards = torch.mean(rewards, -1)[:self.jumps+1].detach().cpu().numpy()
        value_errors = np.mean(value_errors, -1)
        reward_errors = np.mean(reward_errors, -1)

        if trackers:
            update_trackers(trackers, reward_losses, nce_losses, nce_accs,
                            policy_losses, value_losses, total_losses,
                            value_errors, reward_errors, mean_values,
                            mean_rewards, target_values, target_rewards,
                            pred_entropies, target_entropies)

        self.total_steps += 1
        self.update_target_network(self.total_steps)

        return loss


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class SmallEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_size = args.hidden_size
        self.input_channels = 1 if args.grayscale else 3
        self.args = args
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(self.input_channels, 32, 8, stride=2, padding=3)),  # 48x48
            nn.ReLU(),
            nn.BatchNorm2d(32),
            init_(nn.Conv2d(32, 64, 4, stride=2, padding=1)),  # 24x24
            nn.ReLU(),
            nn.BatchNorm2d(64),
            init_(nn.Conv2d(64, 128, 4, stride=2, padding=1)),  # 12 x 12
            nn.ReLU(),
            nn.BatchNorm2d(128),
            init_(nn.Conv2d(128, self.feature_size, 4, stride=2, padding=1)),  # 6 x 6
            nn.ReLU(),
            init_(nn.Conv2d(self.feature_size, self.feature_size, 1, stride=1, padding=0)),
            nn.ReLU())
        self.train()

    def forward(self, inputs):
        fmaps = self.main(inputs)
        return fmaps


class TransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 latent_size=36,
                 action_dim=6,):
        super().__init__()
        self.hidden_size = hidden_size
        self.args = args
        layers = [Conv2dSame(channels+action_dim, hidden_size, 3),
                  nn.ReLU(),
                  nn.BatchNorm2d(hidden_size)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size, hidden_size))
        layers.extend([Conv2dSame(hidden_size, channels, 3),
                      nn.ReLU()])

        self.action_embedding = nn.Embedding(num_actions, latent_size*action_dim)

        self.network = nn.Sequential(*layers)
        self.reward_predictor = ValueNetwork(channels)
        self.train()

    def _make_layer(self, in_channels, depth):
        return nn.Sequential(
            Conv2dSame(in_channels, depth, 3),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth)
        )

    def forward(self, x, action):
        action_embedding = self.action_embedding(action).view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        stacked_image = torch.cat([x, action_embedding], 1)
        next_state = self.network(stacked_image)
        next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward


class ConvFiLM(nn.Module):
    def __init__(self, input_dim, cond_dim, bn=False, one_hot=True):
        super().__init__()
        if one_hot:
            self.embedding = nn.Embedding(cond_dim, cond_dim)
        else:
            self.embedding = nn.Identity()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.conditioning = nn.Linear(cond_dim, input_dim * 2)
        self.bn = nn.BatchNorm2d(input_dim, affine=False) if bn else nn.Identity()

    def forward(self, input, cond):
        cond = self.embedding(cond)
        conditioning = self.conditioning(cond)
        gamma = conditioning[..., :self.input_dim, None, None]
        beta = conditioning[..., self.input_dim:, None, None]
        input = self.bn(input)

        return input * gamma + beta


def renormalize(tensor, first_dim=1):
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)


class FiLMTransitionModel(nn.Module):
    def __init__(self, channels, cond_size, args, blocks=16, hidden_size=256,):
        super().__init__()
        self.hidden_size = hidden_size
        self.args = args
        layers = nn.ModuleList()
        layers.append(Conv2dSame(channels, hidden_size, 3))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(hidden_size))
        for _ in range(blocks):
            layers.append(FiLMResidualBlock(hidden_size, hidden_size, cond_size))
        layers.extend([Conv2dSame(hidden_size, channels, 3),
                      nn.ReLU()])

        self.network = nn.Sequential(*layers)
        self.reward_predictor = ValueNetwork(channels)
        self.train()

    def forward(self, x, action):
        action = action.view(x.shape[0],)
        x = self.network[:3](x)
        for resblock in self.network[3:-2]:
            x = resblock(x, action)
        next_state = self.network[-1](x)
        next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_size):
        super().__init__()
        self.film = ConvFiLM(out_channels, cond_size, bn=True)
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            Conv2dSame(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, a):
        residual = x
        out = self.film(x, a)
        out = self.block(out)
        out += residual
        out = F.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            Conv2dSame(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class QNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_size=128, pixels=36, limit=300):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(hidden_size),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 512),
                  nn.ReLU(),
                  init_small(nn.Linear(512, num_actions*(limit*2 + 1)))]
        self.network = nn.Sequential(*layers)
        self.num_actions = num_actions
        self.dist_size = limit*2 + 1
        self.train()

    def forward(self, x):
        distributions = self.network(x).view(*(x.shape[:-3]),
                                             self.num_actions,
                                             self.dist_size)
        return distributions


class ValueNetwork(nn.Module):
    def __init__(self, input_channels, hidden_size=128, pixels=36, limit=300):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(hidden_size),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 256),
                  nn.ReLU(),
                  init_small(nn.Linear(256, limit*2 + 1))]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


class PolicyNetwork(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_size=128, pixels=36):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [Conv2dSame(input_channels, hidden_size, 3),
                  nn.ReLU(),
                  nn.BatchNorm2d(hidden_size),
                  nn.Flatten(-3, -1),
                  init_small(nn.Linear(pixels * hidden_size, num_actions))]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


class RepNet(nn.Module):
    def __init__(self, framestack=32, grayscale=False, actions=True):
        super().__init__()
        self.input_channels = framestack * (1 if grayscale else 3)
        self.actions = actions
        if self.actions:
            self.input_channels += framestack
        layers = nn.ModuleList()
        hidden_channels = 128
        layers.append(nn.Conv2d(self.input_channels, hidden_channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(hidden_channels))
        for _ in range(2):
            layers.append(ResidualBlock(hidden_channels, hidden_channels))
        layers.append(nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1))
        hidden_channels = hidden_channels * 2
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(hidden_channels))
        for _ in range(3):
            layers.append(ResidualBlock(hidden_channels, hidden_channels))
        layers.append(nn.AvgPool2d(2))
        for _ in range(3):
            layers.append(ResidualBlock(hidden_channels, hidden_channels))
        layers.append(nn.AvgPool2d(2))
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x, actions=None):
        if self.actions:
            actions = actions[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
            stacked_image = torch.cat([x, actions], 1)
        else:
            stacked_image = x
        latent = self.network(stacked_image)
        return renormalize(latent, 1)

    def conv_out_size(self, h, w):
        return (6, 6)


def transform(value, eps=0.001):
    value = value.float()  # Avoid any fp16 shenanigans
    value = torch.sign(value) * (torch.sqrt(torch.abs(value) + 1) - 1) + eps * value
    return value


def inverse_transform(value, eps=0.001):
    value = value.float()  # Avoid any fp16 shenanigans
    return torch.sign(value)*(((torch.sqrt(1+4*eps*(torch.abs(value) + 1 + eps)) - 1)/(2*eps))**2 - 1)


def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit*2+1), device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


def from_categorical(distribution, limit=300, logits=True):
    distribution = distribution.float()  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    weights = torch.arange(-limit, limit + 1, device=distribution.device).float()
    return distribution @ weights


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None
