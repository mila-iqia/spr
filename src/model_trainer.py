import collections

from torch import nn
import torch
from torch.nn import functional as F

from rlpyt.utils.collections import namedarraytuple
from src.envs import get_example_outputs
from src.mcts_memory import ReplayMemory, AsyncPrioritizedSequenceReplayFrameBufferExtended
import numpy as np
from statistics import mean
import wandb

NetworkOutput = namedarraytuple('NetworkOutput', ['next_state', 'reward', 'policy_logits', 'value'])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done", "policy_probs", "value"])


class TrainingWorker(object):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.args = args
        self.initialize_replay_buffer()
        self.maximum_length = args.jumps
        self.multistep = args.multistep
        self.use_all_targets = args.use_all_targets
        self.nce = LocalNCE()
        self.epochs_till_now = 0

        self.train_trackers = dict()
        self.val_trackers = dict()
        self.reset_trackers("train")
        self.reset_trackers("val")

    def initialize_replay_buffer(self):
        examples = get_example_outputs(self.args)
        batch_size = self.args.num_envs
        if self.args.reanalyze:
            batch_size *= 5
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            policy_probs=examples['policy_probs'],
            value=examples['value']
        )
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.args.buffer_size,
            B=batch_size,
            batch_T=self.args.jumps+self.args.multistep, # We don't use the built-in n-step returns, so easiest to just ask for all the data at once.
            rnn_state_interval=0,
            discount=self.args.discount,
            n_step_return=1,
            alpha=self.args.priority_exponent,
            beta=self.args.priority_weight,
            default_priority=1
        )
        self.buffer = AsyncPrioritizedSequenceReplayFrameBufferExtended(**replay_kwargs)

    def samples_to_buffer(self, observation, action, reward, done, policy_probs, value):
        return SamplesToBuffer(
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            policy_probs=policy_probs,
            value=value
        )

    def reset_trackers(self, mode="train"):
        if mode == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers
        trackers["epoch_losses"] = np.zeros(self.maximum_length+1)
        trackers["value_losses"] = np.zeros(self.maximum_length+1)
        trackers["policy_losses"] = np.zeros(self.maximum_length+1)
        trackers["reward_losses"] = np.zeros(self.maximum_length+1)
        trackers["local_losses"] = np.zeros(self.maximum_length+1)
        trackers["value_errors"] = np.zeros(self.maximum_length+1)
        trackers["reward_errors"] = np.zeros(self.maximum_length+1)
        trackers["iterations"] = 0

    def step(self):

        total_losses, reward_losses,\
        contrastive_losses, policy_losses,\
        value_losses, value_errors,\
        reward_errors = self.train()

        self.update_trackers(reward_losses, contrastive_losses, policy_losses,
                             value_losses, total_losses,
                             value_errors,
                             reward_errors)

    def update_trackers(self,
                        reward_losses,
                        local_losses,
                        policy_losses,
                        value_losses,
                        epoch_losses,
                        value_errors,
                        reward_errors,
                        mode="train"):
        if mode == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers

        trackers["iterations"] += 1
        trackers["local_losses"] += np.array(local_losses)
        trackers["reward_losses"] += np.array(reward_losses)
        trackers["policy_losses"] += np.array(policy_losses)
        trackers["value_losses"] += np.array(value_losses)
        trackers["epoch_losses"] += np.array(epoch_losses)
        trackers["value_errors"] += np.array(value_errors)
        trackers["reward_errors"] += np.array(reward_errors)

        return local_losses, reward_losses, value_losses, policy_losses,\
               epoch_losses, value_errors, reward_errors

    def summarize_trackers(self, mode="train"):
        if mode == "train":
            trackers = self.train_trackers
        else:
            trackers = self.val_trackers

        iterations = trackers["iterations"]
        local_losses = np.array(trackers["local_losses"]/iterations)
        reward_losses = np.array(trackers["reward_losses"]/iterations)
        policy_losses = np.array(trackers["policy_losses"]/iterations)
        value_losses = np.array(trackers["value_losses"]/iterations)
        epoch_losses = np.array(trackers["epoch_losses"]/iterations)
        value_errors = np.array(trackers["value_errors"]/iterations)
        reward_errors = np.array(trackers["reward_errors"]/iterations)

        return local_losses, reward_losses, value_losses, policy_losses,\
               epoch_losses, value_errors, reward_errors

    def log_results(self,
                    prefix='train',
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

        local_losses, reward_losses, value_losses, policy_losses, epoch_losses,\
            value_errors, reward_errors = self.summarize_trackers(prefix)
        self.reset_trackers(prefix)
        print(
            "{} Epoch: {}, Epoch Loss: {:.3f}, Local Loss: {:.3f}, Rew. Loss: {:.3f}, Policy Loss: {:.3f}, Value Loss: {:.3f}, Reward Error: {:.3f}, Value Error: {:.3f}".format(
                prefix.capitalize(),
                self.epochs_till_now,
                np.mean(epoch_losses),
                np.mean(local_losses),
                np.mean(reward_losses),
                np.mean(policy_losses),
                np.mean(value_losses),
                np.mean(reward_errors),
                np.mean(value_errors)))

        for i in range(self.maximum_length + 1):
            jump = i
            if verbose_print:
                print(
                    "{} Jump: {}, Epoch Loss: {:.3f}, Local Loss: {:.3f}, Rew. Loss: {:.3f}, Policy Loss: {:.3f}, Value Loss: {:.3f}, Reward Error: {:.3f}, Value Error: {:.3f}".format(
                        prefix.capitalize(),
                        jump,
                        epoch_losses[i],
                        local_losses[i],
                        reward_losses[i],
                        policy_losses[i],
                        value_losses[i],
                        reward_errors[i],
                        value_errors[i]))

            wandb.log({prefix + 'Jump {} loss'.format(jump): epoch_losses[i],
                       prefix + 'Jump {} local loss'.format(jump): local_losses[i],
                       prefix + "Jump {} Reward Loss".format(jump): reward_losses[i],
                       prefix + 'Jump {} Value Loss'.format(jump): value_losses[i],
                       prefix + "Jump {} Reward Error".format(jump): reward_errors[i],
                       prefix + "Jump {} Policy loss".format(jump): policy_losses[i],
                       prefix + "Jump {} Value Error".format(jump): value_errors[i],
                       'FM epoch': self.epochs_till_now})

        wandb.log({prefix + ' loss': np.mean(epoch_losses),
                   prefix + ' local loss': np.mean(local_losses),
                   prefix + " Reward Loss": np.mean(reward_losses),
                   prefix + ' Value Loss': np.mean(value_losses),
                   prefix + " Reward Error": np.mean(reward_errors),
                   prefix + " Policy loss": np.mean(policy_losses),
                   prefix + " Value Error": np.mean(value_errors),
                   'FM epoch': self.epochs_till_now})

    def train(self, step=True):
        """
        Do one update of the model on data drawn from a buffer.
        :param step: whether or not to actually take a gradient step.
        :return: Updated weights for prioritized experience replay.
        """
        with torch.no_grad():
            states, actions, rewards, return_, done, done_n, unk, \
            is_weights, policies, values = self.buffer.sample_batch(self.args.batch_size)

            states = states.float().to(self.args.device)
            actions = actions.long().to(self.args.device)
            rewards = rewards.float().to(self.args.device)
            policies = torch.from_numpy(policies).float().to(self.args.device)
            values = torch.from_numpy(values).float().to(self.args.device)
            initial_states = states[0]
            initial_actions = actions[0]
            is_weights = is_weights.to(self.args.device)

            target_images = states[:, :, 0].transpose(0, 1)
            target_images = target_images.reshape(-1, *states.shape[-3:])

        # Get into the shape used by the NCE code.
        if not self.args.no_nce:
            target_images = self.model.target_encoder(target_images)
            target_images = target_images.view(states.shape[1], -1, *target_images.shape[1:])
            target_images = target_images.flatten(3, 4).permute(3, 0, 1, 2)

        current_state, pred_reward,\
        pred_policy, pred_value = self.model.initial_inference(initial_states,
                                                               initial_actions,
                                                               logits=True)

        predictions = [(1.0, pred_reward, pred_policy, pred_value)]
        pred_states = [current_state]

        pred_values, pred_rewards, pred_policies = [], [], []
        loss = torch.zeros(1, device=self.args.device)
        reward_losses, value_losses, policy_losses, nce_losses, total_losses, value_targets = [], [], [], [], [], []

        discounts = torch.ones_like(rewards)[:self.multistep]*self.args.discount
        discounts = discounts ** torch.arange(0, self.multistep, device=self.args.device)[:, None].float()

        value_errors, reward_errors = [], []

        for i in range(self.maximum_length):
            action = actions[i]
            current_state, pred_reward, pred_policy, pred_value = self.model(current_state, action)

            current_state = ScaleGradient.apply(current_state, 0.5)

            predictions.append((1. / self.maximum_length,
                                pred_reward,
                                pred_policy,
                                pred_value))
            pred_states.append(current_state)

        for i, prediction in enumerate(predictions):

            loss_scale, pred_reward, pred_policy, pred_value = prediction
            value_target = torch.sum(discounts*rewards[i:i+self.multistep], 0)
            value_target = value_target + self.args.discount ** self.multistep \
                           * values[i+self.multistep]
            value_targets.append(value_target)
            value_target = to_categorical(transform(value_target))
            reward_target = to_categorical(transform(rewards[i]))

            pred_rewards.append(inverse_transform(from_categorical(pred_reward, logits=True)))
            pred_values.append(inverse_transform(from_categorical(pred_value, logits=True)))
            pred_policies.append(pred_policy)

            value_errors.append(torch.abs(pred_values[-1] - values[i]).detach().cpu().numpy())
            reward_errors.append(torch.abs(pred_rewards[-1] - rewards[i]).detach().cpu().numpy())

            pred_value = F.log_softmax(pred_value, -1)
            pred_reward = F.log_softmax(pred_reward, -1)
            pred_policy = F.log_softmax(pred_policy, -1)

            current_reward_loss = -torch.sum(reward_target * pred_reward, -1).mean()
            current_value_loss = -torch.sum(value_target * pred_value, -1).mean()
            current_policy_loss = -torch.sum(policies[i] * pred_policy, -1).mean()

            reward_losses.append(current_reward_loss.detach().cpu().item())
            value_losses.append(current_value_loss.detach().cpu().item())
            policy_losses.append(current_policy_loss.detach().cpu().item())

            if not self.args.no_nce:
                if self.use_all_targets:
                    current_targets = target_images.roll(-i, 2).flatten(1, 2)
                else:
                    current_targets = target_images[:, :, i]
                nce_input = current_state.flatten(2, 3).permute(2, 0, 1)
                current_nce_loss = self.nce(nce_input, current_targets).mean()
                nce_losses.append(current_nce_loss.detach().cpu().item())
            else:
                current_nce_loss = 0
                nce_losses.append(0)

            current_loss = current_value_loss*self.args.value_loss_weight + \
                           current_policy_loss*self.args.policy_loss_weight + \
                           current_reward_loss*self.args.reward_loss_weight + \
                           current_nce_loss*self.args.contrastive_loss_weight

            total_losses.append(current_loss.mean().detach().cpu().item())
            current_loss = (current_loss * is_weights).mean()
            loss = loss + current_loss

        self.buffer.update_batch_priorities(value_errors[0])

        value_errors = np.mean(value_errors, -1)
        reward_errors = np.mean(reward_errors, -1)

        loss = loss.mean()
        if step:
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
            self.model.scheduler.step()

        self.epochs_till_now += 1

        return total_losses, reward_losses, nce_losses, policy_losses,\
               value_losses, value_errors, reward_errors


class LocalNCE(nn.Module):
    def __init__(self):
        super().__init__()

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
        n_batch = f_x1.size(1)
        neg_batch = f_x2.size(1)
        # reshaping for big matrix multiply
        # f_x1 = f_x1.permute(2, 0, 1)  # (n_locs, n_batch, n_rkhs)
        f_x2 = f_x2.permute(0, 2, 1)  # (n_locs, n_rkhs, n_batch)
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
        return loss_nce


class MCTSModel(nn.Module):
    def __init__(self, args, num_actions):
        super().__init__()
        if args.film:
            self.dynamics_model = FiLMTransitionModel(args.hidden_size,
                                                      num_actions,
                                                      args.dynamics_blocks)
        else:
            self.dynamics_model = TransitionModel(args.hidden_size,
                                                  num_actions,
                                                  args.dynamics_blocks)
        self.value_model = ValueNetwork(args.hidden_size)
        self.policy_model = PolicyNetwork(args.hidden_size, num_actions)
        self.encoder = RepNet(args.framestack, grayscale=args.grayscale, actions=False)
        self.target_encoder = SmallEncoder(args)

        params = list(self.dynamics_model.parameters()) + \
            list(self.value_model.parameters()) +\
            list(self.policy_model.parameters()) +\
            list(self.encoder.parameters()) +\
            list(self.target_encoder.parameters())
        self.optimizer = torch.optim.SGD(params,
                                         lr=args.learning_rate,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                args.lr_decay ** (1. / args.lr_decay_steps), )

    def encode(self, images, actions):
        return self.encoder(images, actions)

    def initial_inference(self, obs, actions=None, logits=False):
        if len(obs.shape) < 5:
            obs = obs.unsqueeze(0)
        obs = obs.flatten(1, 2)
        hidden_state = self.encoder(obs, actions)
        policy_logits = self.policy_model(hidden_state)
        value_logits = self.value_model(hidden_state)
        reward_logits = self.dynamics_model.reward_predictor(hidden_state)

        if logits:
            return NetworkOutput(hidden_state, reward_logits, policy_logits, value_logits)

        value = inverse_transform(from_categorical(value_logits, logits=True))
        reward = inverse_transform(from_categorical(reward_logits, logits=True))
        return NetworkOutput(hidden_state, reward, policy_logits, value)

    def inference(self, state, action):
        next_state, reward_logits, policy_logits, value_logits = self.forward(state, action)
        value = inverse_transform(from_categorical(value_logits, logits=True))
        reward = inverse_transform(from_categorical(reward_logits, logits=True))

        return NetworkOutput(next_state, reward, policy_logits, value)

    def forward(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        policy_logits = self.policy_model(next_state)
        value_logits = self.value_model(next_state)

        return next_state, reward_logits, policy_logits, value_logits


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
            nn.ReLU())
        self.train()

    def forward(self, inputs):
        fmaps = self.main(inputs)
        return fmaps


class TransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 blocks=16,
                 hidden_size=256,
                 latent_size=36,
                 action_dim=6):
        super().__init__()
        self.hidden_size = hidden_size
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
    def __init__(self, input_channels, cond_size, blocks=16, hidden_size=256, output_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        layers = nn.ModuleList()
        layers.append(Conv2dSame(input_channels, hidden_size, 3))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(hidden_size))
        for _ in range(blocks):
            layers.append(FiLMResidualBlock(hidden_size, hidden_size, cond_size))
        layers.extend([Conv2dSame(hidden_size, output_size, 3),
                      nn.ReLU()])

        self.network = nn.Sequential(*layers)
        self.reward_predictor = ValueNetwork(output_size)
        self.train()

    def forward(self, x, action):
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
                  nn.Linear(256, limit*2 + 1)]
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
                  nn.Linear(pixels*hidden_size, num_actions)]
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
        return self.network(stacked_image)


def transform(value, eps=0.001):
    value = torch.sign(value) * (torch.sqrt(torch.abs(value) + 1) - 1 + eps * value)
    return value


def inverse_transform(value, eps=0.001):
    return torch.sign(value) * eps ** -2 * 2 * (1 + 2 * eps + 2 * eps * torch.abs(value) - torch.sqrt(
        1 + 4 * eps + 4 * eps ** 2 + 4 * eps * torch.abs(value)))


def to_categorical(value, limit=300):
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
