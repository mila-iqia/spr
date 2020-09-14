import torch
import numpy as np
import torch.nn as nn

from rlpyt.models.mlp import MlpModel
from src.rlpyt_models import Conv2dModel


class MCTSModel(torch.nn.Module):
    def __init__(self, image_shape, output_size, jumps, mpr, imagesize,
                 augmentation, target_augmentation, eval_augmentation,
                 dynamics_blocks, norm_type, dqn_hidden_size, model_rl,
                 momentum_tau, renormalize, dropout, residual_tm):
        f, c = image_shape[:2]
        in_channels = np.prod(image_shape[:2])
        self.conv = Conv2dModel(
            in_channels=in_channels,
            channels=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[0, 0, 0],
            use_maxpool=False,
            dropout=dropout,
        )

        fake_input = torch.zeros(1, f*c, imagesize, imagesize)
        fake_output = self.conv(fake_input)
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        self.jumps = jumps
        self.model_rl = model_rl
        self.use_mpr = mpr
        self.target_augmentation = target_augmentation
        self.eval_augmentation = eval_augmentation
        self.num_actions = output_size

        self.head = MLPHead(input_channels=self.hidden_size, pixels=self.pixels)

        if self.jumps > 0:
            self.dynamics_model = TransitionModel(channels=self.hidden_size,
                                                  num_actions=output_size,
                                                  pixels=self.pixels,
                                                  hidden_size=self.hidden_size,
                                                  limit=1,
                                                  blocks=dynamics_blocks,
                                                  norm_type=norm_type,
                                                  renormalize=renormalize,
                                                  residual=residual_tm)
        else:
            self.dynamics_model = nn.Identity()

        self.renormalize = renormalize


class MLPHead(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_size=1,
                 hidden_size=256,
                 pixels=30,
                 noisy=0):
        super().__init__()
        if noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear
        self.noisy = noisy
        self.linears = [linear(input_channels*pixels, hidden_size),
                        linear(hidden_size, output_size)]
        layers = [nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        self._output_size = output_size

    def forward(self, input):
        return self.network(input)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class TransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 norm_type="bn",
                 renormalize=True,
                 residual=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize
        self.residual = residual
        layers = [Conv2dSame(channels+num_actions, hidden_size, 3),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size,
                                        norm_type))
        layers.extend([Conv2dSame(hidden_size, channels, 3)])

        self.action_embedding = nn.Embedding(num_actions, pixels*action_dim)

        self.network = nn.Sequential(*layers)
        self.reward_predictor = RewardPredictor(channels,
                                                pixels=pixels,
                                                limit=limit,
                                                norm_type=norm_type)
        self.train()

    def forward(self, x, action):
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        stacked_image = torch.cat([x, action_onehot], 1)
        next_state = self.network(stacked_image)
        if self.residual:
            next_state = next_state + x
        next_state = F.relu(next_state)
        if self.renormalize:
            next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type="bn"):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            init_normalization(out_channels, norm_type),
            Conv2dSame(out_channels, out_channels, 3),
            init_normalization(out_channels, norm_type),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class Conv2dSame(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                            stride=stride, padding=ka)
        )

    def forward(self, x):
        return self.net(x)


def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "none" or type is None:
        return nn.Identity()


class RewardPredictor(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_size=1,
                 pixels=36,
                 limit=300,
                 norm_type="bn"):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 256),
                  nn.ReLU(),
                  nn.Linear(256, limit*2 + 1)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)



