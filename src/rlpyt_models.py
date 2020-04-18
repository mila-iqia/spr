import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose, RandomCrop,\
    ToPILImage, ToTensor, RandomChoice, RandomAffine, CenterCrop

from rlpyt.models.dqn.atari_catdqn_model import DistributionalHeadModel
from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel
from rlpyt.models.utils import scale_grad, update_state_dict
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from src.model_trainer import ValueNetwork, TransitionModel, \
    NetworkOutput, from_categorical, ScaleGradient, BlockNCE, init, \
    ResidualBlock, renormalize, FiLMTransitionModel, init_normalization
import numpy as np
from skimage.util import view_as_windows
from rlpyt.utils.logging import logger
import wandb
import time


class AsyncRlEvalWandb(AsyncRlEval):
    def log_diagnostics(self, itr, sampler_itr, throttle_time):
        cum_steps = sampler_itr * self.sampler.batch_size
        self.wandb_info = {'cum_steps': cum_steps}
        super().log_diagnostics(itr, sampler_itr, throttle_time)
        wandb.log(self.wandb_info)

    def _log_infos(self, traj_infos=None):
        """
        Writes trajectory info and optimizer info into csv via the logger.
        Resets stored optimizer info.
        """
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    values = [info[k] for info in traj_infos]
                    logger.record_tabular_misc_stat(k,
                                                    values)
                    self.wandb_info[k + "Average"] = np.average(values)
                    self.wandb_info[k + "Median"] = np.median(values)

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
                self.wandb_info[k] = np.average(v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


class MinibatchRlEvalWandb(MinibatchRlEval):
    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size
        self.wandb_info = {'cum_steps': cum_steps}
        super().log_diagnostics(itr, eval_traj_infos, eval_time)
        wandb.log(self.wandb_info)

    def _log_infos(self, traj_infos=None):
        """
        Writes trajectory info and optimizer info into csv via the logger.
        Resets stored optimizer info.
        """
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    values = [info[k] for info in traj_infos]
                    logger.record_tabular_misc_stat(k,
                                                    values)
                    self.wandb_info[k + "Average"] = np.average(values)
                    self.wandb_info[k + "Median"] = np.median(values)

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
                self.wandb_info[k] = np.average(v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)

class PizeroCatDqnModel(torch.nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
            self,
            image_shape,
            output_size,
            n_atoms=51,
            fc_sizes=512,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            framestack=4,
            grayscale=True,
            actions=False,
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling
        f, c, h, w = image_shape
        self.conv = RepNet(f*c)
        # conv_out_size = self.conv.conv_out_size(h, w)
        # self.dyamics_network = TransitionModel(conv_out_size, num_actions)
        # self.reward_network = ValueNetwork(conv_out_size)
        if dueling:
            self.head = PizeroDistributionalDuelingHeadModel(256, output_size, pixels=36)
        else:
            self.head = PizeroDistributionalHeadModel(256, output_size, pixels=36)

    def forward(self, observation, prev_action, prev_reward):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        while len(observation.shape) <= 4:
            observation = observation.unsqueeze(0)
        observation = observation.flatten(-4, -3)
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        p = self.head(conv_out)
        p = F.softmax(p, dim=-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p.squeeze()


class PizeroSearchCatDqnModel(torch.nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
            self,
            image_shape,
            output_size,
            n_atoms=51,
            fc_sizes=512,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            framestack=4,
            grayscale=True,
            actions=False,
            jumps=0,
            detach_model=True,
            nce=False,
            augmentation=False,
            stack_actions=False,
            dynamics_blocks=16,
            film=False,
            norm_type="bn"
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling
        f, c, h, w = image_shape
        self.conv = RepNet(f*c, norm_type=norm_type)
        self.jumps = jumps
        self.detach_model = detach_model
        self.nce = nce
        self.augmentation = augmentation
        self.stack_actions = stack_actions
        self.pixels = 25 if self.augmentation else 36
        if dueling:
            self.head = PizeroDistributionalDuelingHeadModel(256, output_size,
                                                             pixels=self.pixels,
                                                             norm_type=norm_type)
        else:
            self.head = PizeroDistributionalHeadModel(256, output_size,
                                                      pixels=self.pixels,
                                                      norm_type=norm_type)
            
        if film:
            dynamics_model = FiLMTransitionModel
        else:
            dynamics_model = TransitionModel

        self.dynamics_model = dynamics_model(channels=256,
                                             num_actions=output_size,
                                             pixels=self.pixels,
                                             limit=1,
                                             blocks=dynamics_blocks,
                                             norm_type=norm_type)

        if self.nce:
            if self.stack_actions:
                input_size = c - 1
            else:
                input_size = c
            self.nce_target_encoder = SmallEncoder(256, input_size,
                                                   norm_type=norm_type)
            self.classifier = nn.Sequential(nn.Linear(256, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 256),
                                            nn.ReLU())
            self.nce = BlockNCE(self.classifier,
                                use_self_targets=False)

        if self.detach_model:
            if dueling:
                self.target_head = PizeroDistributionalDuelingHeadModel(256, output_size,
                                                                        pixels=self.pixels,
                                                                        norm_type=norm_type)
            else:
                self.target_head = PizeroDistributionalHeadModel(256, output_size,
                                                                 pixels=self.pixels,
                                                                 norm_type=norm_type)

            for param in self.target_head.parameters():
                param.requires_grad = False

        # if self.augmentation:
        #     transforms = list()
        #     transforms.append(ToPILImage())
        #     transforms.append(RandomCrop((84, 84)))
        #     transforms.append(ToTensor())
        #     self.transforms = Compose(transforms)
        #     eval_transforms = list()
        #     eval_transforms.append(ToPILImage())
        #     eval_transforms.append(CenterCrop((84, 84)))
        #     eval_transforms.append(ToTensor())
        #     self.eval_transforms = Compose(eval_transforms)

    def transform(self, images, eval=False):
        images = images.float()/255. if images.dtype == torch.uint8 else images
        if not self.augmentation:
            return images
        flat_images = images.view(-1, *images.shape[-3:])
        if eval:
            processed_images = flat_images[:, :, 8:-8, 8:-8]
        else:
            flat_images = flat_images.cpu().numpy()
            processed_images = curl_random_crop(flat_images, 84)
            processed_images = torch.from_numpy(processed_images).to(images.device)

        processed_images.view(*images.shape[:-3], *processed_images.shape[1:])
        #
        # transforms = self.eval_transforms if eval else self.transforms
        # if eval:
        #
        # processed_images = [transforms(i.cpu()) for i in flat_images]
        # processed_images = torch.stack(processed_images).to(images.device)
        # processed_images.view(*images.shape[:-3], *processed_images.shape[1:])
        return processed_images

    def stem_parameters(self):
        return list(self.conv.parameters()) + list(self.head.parameters())

    def stem_forward(self, img, prev_action, prev_reward):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        return conv_out

    def head_forward(self, conv_out, prev_action, prev_reward, target=False):
        lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)
        if target:
            p = self.target_head(conv_out)
        else:
            p = self.head(conv_out)
        p = F.softmax(p, dim=-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    def forward(self, observation, prev_action, prev_reward, jumps=False):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        # start = time.time()
        if jumps:
            pred_ps = []
            pred_reward = []
            pred_latents = []
            input_obs = observation[0].flatten(1, 2)
            input_obs = self.transform(input_obs)
            latent = self.stem_forward(input_obs,
                                       prev_action[0],
                                       prev_reward[0])
            pred_ps.append(self.head_forward(latent,
                                             prev_action[0],
                                             prev_reward[0]),)

            pred_latents.append(latent)


            if self.detach_model and self.jumps > 0:
                # copy_start = time.time()
                self.target_head.load_state_dict(self.head.state_dict())
                # copy_end = time.time()
                # print("Copying took {}".format(copy_end - copy_start))
                latent = latent.detach()

            pred_rew = self.dynamics_model.reward_predictor(latent)
            pred_reward.append(pred_rew)

            for j in range(1, self.jumps + 1):
                latent, pred_rew, _, _ = self.step(latent, prev_action[j])
                latent = ScaleGradient.apply(latent, 0.5)
                pred_latents.append(latent)
                pred_reward.append(pred_rew)
                pred_ps.append(self.head_forward(latent,
                                                 prev_action[j],
                                                 prev_reward[j],
                                                 target=self.detach_model))

            if self.nce:
                if self.stack_actions:
                    observation = observation[:, :, :, :-1]
                if self.jumps > 0:
                    target_images = observation[0:self.jumps + 1, :, -1].transpose(0, 1)
                else:
                    target_images = observation[1, :, -1].transpose(0, 1)
                target_images = self.transform(target_images)
                if len(target_images.shape) == 4:
                    target_images = target_images.unsqueeze(2)
                target_latents = self.nce_target_encoder(target_images.flatten(0, 1))
                target_latents = target_latents.view(observation.shape[1], -1,
                                                     *target_latents.shape[1:])
                target_latents = target_latents.flatten(3, 4).permute(3, 0, 1, 2)
                target_latents = target_latents.permute(0, 2, 1, 3)
                nce_input = torch.stack(pred_latents, 1).flatten(3, 4).permute(3, 1, 0, 2)
                nce_loss, nce_accs = self.nce.forward(nce_input, target_latents)
                nce_loss = nce_loss.mean(0)
                nce_accs = nce_accs.mean()

            else:
                nce_loss = 0
                nce_accs = 0

            # end = time.time()
            # print("Forward took {}".format(end - start))
            return pred_ps,\
                   [F.log_softmax(ps, -1) for ps in pred_reward],\
                   nce_loss, nce_accs

        else:
            # img = observation.type(torch.float)  # Expect torch.uint8 inputs
            # img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.
            observation = observation.flatten(-4, -3)
            img = self.transform(observation, True)

            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

            conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
            p = self.head(conv_out)
            p = F.softmax(p, dim=-1)

            # Restore leading dimensions: [T,B], [B], or [], as input.
            p = restore_leading_dims(p, lead_dim, T, B)
            return p

    def initial_inference(self, obs, actions=None, logits=False):
        if len(obs.shape) == 5:
            obs = obs.flatten(1, 2)
        obs = self.transform(obs, True)
        hidden_state = self.conv(obs)
        # if not self.args.q_learning:
        #     policy_logits = self.policy_model(hidden_state)
        # else:
        policy_logits = None
        value_logits = self.head(hidden_state)
        reward_logits = self.dynamics_model.reward_predictor(hidden_state)

        if logits:
            return NetworkOutput(hidden_state, reward_logits, policy_logits, value_logits)

        value = from_categorical(value_logits, logits=True, limit=10) #TODO Make these configurable
        reward = from_categorical(reward_logits, logits=True, limit=1)
        return NetworkOutput(hidden_state, reward, policy_logits, value)

    def inference(self, state, action):
        next_state, reward_logits, \
        policy_logits, value_logits = self.step(state, action)
        value = from_categorical(value_logits, logits=True, limit=10) #TODO Make these configurable
        reward = from_categorical(reward_logits, logits=True, limit=1)

        return NetworkOutput(next_state, reward, policy_logits, value)

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        policy_logits = None
        value_logits = self.head(next_state)
        return next_state, reward_logits, policy_logits, value_logits

class PizeroDistributionalHeadModel(torch.nn.Module):
    """An MLP head which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=128,
                 pixels=30,
                 n_atoms=51,
                 norm_type="bn"):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 512),
                  nn.ReLU(),
                  nn.Linear(512, output_size*n_atoms)]
        self.network = nn.Sequential(*layers)
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        return self.network(input).view(-1, self._output_size, self._n_atoms)


class PizeroDistributionalDuelingHeadModel(torch.nn.Module):
    """An MLP head which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=128,
                 pixels=30,
                 n_atoms=51,
                 grad_scale=2 ** (-1 / 2),
                 norm_type="bn"):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 512),
                  nn.ReLU(),
                  nn.Linear(512, n_atoms)]
        self.advantage_hidden = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
            nn.ReLU(),
            init_normalization(hidden_size, norm_type),
            nn.Flatten(-3, -1))
        self.advantage_out = torch.nn.Linear(pixels*hidden_size,
                                             output_size * n_atoms, bias=False)
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms))
        self.value = nn.Sequential(*layers)
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        x = self.advantage_out(x)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias


class SmallEncoder(nn.Module):
    def __init__(self,
                 feature_size,
                 input_channels,
                 norm_type="bn"):
        super().__init__()
        self.feature_size = feature_size
        self.input_channels = input_channels
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(self.input_channels, 32, 8, stride=2, padding=3)),  # 48x48
            nn.ReLU(),
            init_normalization(32, norm_type),
            init_(nn.Conv2d(32, 64, 4, stride=2, padding=1)),  # 24x24
            nn.ReLU(),
            init_normalization(64, norm_type),
            init_(nn.Conv2d(64, 128, 4, stride=2, padding=1)),  # 12 x 12
            nn.ReLU(),
            init_normalization(128, norm_type),
            init_(nn.Conv2d(128, self.feature_size, 4, stride=2, padding=1)),  # 6 x 6
            nn.ReLU(),
            init_(nn.Conv2d(self.feature_size, self.feature_size,
                            1, stride=1, padding=0)),
            nn.ReLU())
        self.train()

    def forward(self, inputs):
        fmaps = self.main(inputs)
        return fmaps


class RepNet(nn.Module):
    def __init__(self, channels=3, norm_type="bn"):
        super().__init__()
        self.input_channels = channels
        layers = nn.ModuleList()
        hidden_channels = 128
        layers.append(nn.Conv2d(self.input_channels, hidden_channels,
                                kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        layers.append(init_normalization(hidden_channels, norm_type))
        for _ in range(2):
            layers.append(ResidualBlock(hidden_channels,
                                        hidden_channels,
                                        norm_type))
        layers.append(nn.Conv2d(hidden_channels, hidden_channels * 2,
                                kernel_size=3, stride=2, padding=1))
        hidden_channels = hidden_channels * 2
        layers.append(nn.ReLU())
        layers.append(init_normalization(hidden_channels, norm_type))
        for _ in range(3):
            layers.append(ResidualBlock(hidden_channels,
                                        hidden_channels,
                                        norm_type))
        layers.append(nn.AvgPool2d(2))
        for _ in range(3):
            layers.append(ResidualBlock(hidden_channels,
                                        hidden_channels,
                                        norm_type))
        layers.append(nn.AvgPool2d(2))
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        if x.shape[-3] < self.input_channels:
            # We need to consolidate the framestack.
            x = x.flatten(-4, -3)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        latent = self.network(x)
        return renormalize(latent, 1)

def curl_random_crop(imgs, out):
    """
    Vectorized random crop
    args:
    imgs: shape (B,C,H,W)
    out: output size (e.g. 84)
    """
    # n: batch size.
    n = imgs.shape[0]
    img_size = imgs.shape[-1] # e.g. 100
    crop_max = img_size - out
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding window
    # combinations of size (out)
    windows = view_as_windows(imgs, (1, out, out, 1))[..., 0,:,:, 0]
    # selects a random window# for each batch element
    cropped = windows[np.arange(n), w1, h1]
    return cropped
