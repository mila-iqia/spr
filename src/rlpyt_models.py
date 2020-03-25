import torch
import torch.nn.functional as F
import torch.nn as nn

from rlpyt.models.dqn.atari_catdqn_model import DistributionalHeadModel
from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel
from rlpyt.models.utils import scale_grad
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from src.model_trainer import ValueNetwork, TransitionModel, RepNet
import numpy as np
from rlpyt.utils.logging import logger
import wandb


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
        c, h, w = image_shape
        self.conv = RepNet(framestack, grayscale, actions)
        # conv_out_size = self.conv.conv_out_size(h, w)
        # self.dyamics_network = TransitionModel(conv_out_size, num_actions)
        # self.reward_network = ValueNetwork(conv_out_size)
        if dueling:
            self.head = PizeroDistributionalDuelingHeadModel(256, output_size)
        else:
            self.head = PizeroDistributionalHeadModel(256, output_size)

    def forward(self, observation, prev_action, prev_reward):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        p = self.head(conv_out)
        p = F.softmax(p, dim=-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p


class PizeroDistributionalHeadModel(torch.nn.Module):
    """An MLP head which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self, input_channels, output_size, hidden_size=128, pixels=30, n_atoms=51):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(hidden_size),
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

    def __init__(self, input_channels, output_size, hidden_size=128, pixels=30, n_atoms=51, grad_scale=2 ** (-1 / 2)):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(hidden_size),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 512),
                  nn.ReLU(),
                  nn.Linear(512, output_size*n_atoms)]
        self.advantage_hidden = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
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

