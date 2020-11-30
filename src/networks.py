import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from src.utils import renormalize
from rlpyt.models.utils import scale_grad


def fixup_init(layer, num_layers):
    nn.init.normal_(layer.weight, mean=0, std=np.sqrt(
        2 / (layer.weight.shape[0] * np.prod(layer.weight.shape[2:]))) * num_layers ** (-0.25))


class WSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels,
                         kernel_size, stride,
                         padding, dilation,
                         groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 norm_type, num_layers=1, groups=-1,
                 drop_prob=0., bias=True, use_ws=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2, 3]
        self.drop_prob = drop_prob

        hidden_dim = round(in_channels * expand_ratio)

        if groups <= 0:
            groups = hidden_dim

        if use_ws:
            conv = WSConv2d
        else:
            conv = nn.Conv2d

        if stride != 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, stride, stride)
            nn.init.normal_(self.downsample.weight, mean=0, std=
                            np.sqrt(2 / (self.downsample.weight.shape[0] *
                            np.prod(self.downsample.weight.shape[2:]))))
        else:
            self.downsample = False

        if expand_ratio == 1:
            conv1 = conv(hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=bias)
            conv2 = conv(hidden_dim, out_channels, 1, 1, 0, bias=bias)
            if not use_ws:
                fixup_init(conv1, num_layers)
                fixup_init(conv2, num_layers)
            self.conv = nn.Sequential(
                # dw
                conv1,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # pw-linear
                conv2,
                init_normalization(out_channels, norm_type),
            )
            nn.init.constant_(self.conv[-1].weight, 0)
        else:
            conv1 = conv(in_channels, hidden_dim, 1, 1, 0, bias=bias)
            conv2 = conv(hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=bias)
            conv3 = conv(hidden_dim, out_channels, 1, 1, 0, bias=bias)
            if not use_ws:
                fixup_init(conv1, num_layers)
                fixup_init(conv2, num_layers)
                fixup_init(conv3, num_layers)
            self.conv = nn.Sequential(
                # pw
                conv1,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # dw
                conv2,
                init_normalization(hidden_dim, norm_type),
                nn.ReLU(inplace=True),
                # pw-linear
                conv3,
                init_normalization(out_channels, norm_type)
            )
            if norm_type != "none":
                nn.init.constant_(self.conv[-1].weight, 0)

    def forward(self, x):
        if self.downsample:
            identity = self.downsample(x)
        else:
            identity = x
        if self.training and np.random.uniform() < self.drop_prob:
            return identity
        else:
            return identity + self.conv(x)


class Residual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, groups=1)


class ImpalaCNN(nn.Module):
    def __init__(self, input_channels,
                 depths=[16, 32, 64],
                 strides=[3, 2, 2],
                 drop_prob=0,
                 norm_type="bn",
                 resblock=InvertedResidual,
                 expand_ratio=2,
                 use_ws=False,):
        super(ImpalaCNN, self).__init__()
        self.depths = [input_channels] + depths
        self.resblock = resblock
        self.expand_ratio = expand_ratio
        self.layers = []
        self.norm_type = norm_type
        self.drop_prob = drop_prob
        self.num_layers = 3*len(depths)
        self.use_ws = use_ws
        for i in range(len(depths)):
            self.layers.append(self._make_layer(self.depths[i],
                                                self.depths[i+1],
                                                strides[i],
                                                block_id=i))
        self.layers = nn.Sequential(*self.layers)
        self.train()

    def _make_layer(self, in_channels, depth, stride, block_id):
        return nn.Sequential(
            self.resblock(in_channels, depth, expand_ratio=self.expand_ratio, stride=stride, norm_type=self.norm_type,
                          drop_prob=self.drop_prob*(3*block_id+0)/self.num_layers,
                          num_layers=self.num_layers, use_ws=self.use_ws),
            self.resblock(depth, depth, expand_ratio=self.expand_ratio, stride=1, norm_type=self.norm_type,
                          drop_prob=self.drop_prob*(3*block_id+1)/self.num_layers,
                          num_layers=self.num_layers, use_ws=self.use_ws),
            self.resblock(depth, depth, expand_ratio=self.expand_ratio, stride=1, norm_type=self.norm_type,
                          drop_prob=self.drop_prob*(3*block_id+2)/self.num_layers,
                          num_layers=self.num_layers, use_ws=self.use_ws)
        )

    @property
    def local_layer_depth(self):
        return self.depths[-2]

    def forward(self, inputs):
        return self.layers(inputs)

class TransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 norm_type="bn",
                 renormalize=True,
                 drop_prob=0,
                 use_ws=False,
                 resblock=InvertedResidual,
                 expand_ratio=2,
                 residual=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize

        self.residual = residual
        if use_ws:
            conv = WSConv2d
        else:
            conv = nn.Conv2d
        self.initial_layer = nn.Sequential(conv(channels+num_actions, hidden_size, 3, 1, 1),
                                           nn.ReLU(), init_normalization(hidden_size, norm_type))
        # if isinstance(self.renormalize, nn.Module) and not isinstance(self.renormalize, nn.Identity):
        #     self.final_layer = conv(hidden_size, channels, 3, 1, 1)
        # else:
        self.final_layer = nn.Conv2d(hidden_size, channels, 3, 1, 1)
        resblocks = []

        for i in range(blocks):
            resblocks.append(resblock(hidden_size,
                                      hidden_size,
                                      stride=1,
                                      norm_type=norm_type,
                                      expand_ratio=expand_ratio,
                                      use_ws=use_ws,
                                      drop_prob=drop_prob*(i+1)/blocks,
                                      num_layers=blocks))
        self.resnet = nn.Sequential(*resblocks)
        if self.residual:
            nn.init.constant_(self.final_layer.weight, 0)
        self.train()

    def forward(self, x, action, blocks=True):
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        stacked_image = torch.cat([x, action_onehot], 1)
        next_state = self.initial_layer(stacked_image)
        if blocks:
            next_state = self.resnet(next_state)
        next_state = self.final_layer(next_state)
        if self.residual:
            next_state = next_state + x
        next_state = F.relu(next_state)
        next_state = self.renormalize(next_state)
        return next_state


class RewardPredictor(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_size=1,
                 pixels=36,
                 limit=300,
                 norm_type="bn"):
        super().__init__()
        self.hidden_size = hidden_size
        self.needs_actions=False
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


class DonePredictor(nn.Module):
    def __init__(self,
                 input_channels,
                 num_actions,
                 hidden_size=4,
                 pixels=36,
                 norm_type="bn"):
        super().__init__()
        self.hidden_size = hidden_size
        self.needs_actions = True
        self.num_actions = num_actions
        layers = [nn.Conv2d(input_channels+num_actions, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 256),
                  nn.ReLU(),
                  nn.Linear(256, 1),
                  nn.Sigmoid()]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x, action):
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        x = torch.cat([x, action_onehot], 1)
        return self.network(x).squeeze(-1)


def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "gn", "max", "none", None]
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
    elif type == "gn":
        groups = max(min(32, channels//4), 1)
        return nn.GroupNorm(groups, channels, affine=affine)
    elif type == "max":
        if not one_d:
            return renormalize
        else:
            return lambda x: renormalize(x, -1)
    elif type == "none" or type is None:
        return nn.Identity()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, bias=True):
        super(NoisyLinear, self).__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.sampling = True
        self.noise_override = None
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('old_bias_epsilon', torch.empty(out_features))
        self.register_buffer('old_weight_epsilon', torch.empty(out_features, in_features))
        self.reset_parameters()
        self.reset_noise()
        self.use_old_noise = False

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        if not self.bias:
            self.bias_mu.fill_(0)
            self.bias_sigma.fill_(0)
        else:
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_mu.data.uniform_(-mu_range, mu_range)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        self.old_bias_epsilon.copy_(self.bias_epsilon)
        self.old_weight_epsilon.copy_(self.weight_epsilon)
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        # Self.training alone isn't a good-enough check, since we may need to
        # activate .eval() during sampling even when we want to use noise
        # (due to batchnorm, dropout, or similar).
        # The extra "sampling" flag serves to override this behavior and causes
        # noise to be used even when .eval() has been called.
        use_noise = (self.training or self.sampling) if self.noise_override is None else self.noise_override
        if use_noise:
            weight_eps = self.old_weight_epsilon if self.use_old_noise else self.weight_epsilon
            bias_eps = self.old_bias_epsilon if self.use_old_noise else self.bias_epsilon

            return F.linear(input, self.weight_mu + self.weight_sigma * weight_eps,
                            self.bias_mu + self.bias_sigma * bias_eps)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            dropout=0.,
            norm_type="none",
            use_ws=False,
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        assert not (use_ws and norm_type == "none")
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride, oc in zip(conv_layers, maxp_strides, channels):
            sequence.extend([conv_layer, init_normalization(oc, norm_type), nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)


class DQNDistributionalDuelingHeadModel(torch.nn.Module):
    """An MLP head with optional noisy layers which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 pixels=30,
                 n_atoms=51,
                 hidden_size=256,
                 grad_scale=2 ** (-1 / 2),
                 noisy=0,
                 std_init=0.1):
        super().__init__()
        if noisy:
            self.linears = [NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, output_size * n_atoms, std_init=std_init),
                            NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, n_atoms, std_init=std_init)
                            ]
        else:
            self.linears = [nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, output_size * n_atoms),
                            nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, n_atoms)
                            ]
        self.advantage_layers = [nn.Flatten(-3, -1),
                                 self.linears[0],
                                 nn.ReLU(),
                                 self.linears[1]]
        self.value_layers = [nn.Flatten(-3, -1),
                             self.linears[2],
                             nn.ReLU(),
                             self.linears[3]]
        self.advantage_net = nn.Sequential(*self.advantage_layers)
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.value_net = nn.Sequential(*self.value_layers)
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input, noise_override=None, old_noise=False):
        [setattr(module, "use_old_noise", old_noise) for module in self.modules()]
        [setattr(module, "noise_override", noise_override) for module in self.modules()]
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value_net(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_net(input)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cosines=64, embedding_dim=7*7*64, noisy_net=False):
        super(CosineEmbeddingNetwork, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear

        self.net = nn.Sequential(
            linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus, states):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
            ).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)

        # uh-oh, somebody has expanded states along the batch dimension,
        # probably to do rollouts in parallel.  Since the standard in this
        # codebase is to expand on dimension 1 and then flatten 0 and 1, we can
        # do the same here to get the right shape.
        if states.shape[0] != batch_size:
            expansion_factor = states.shape[0]//batch_size
            tau_embeddings = tau_embeddings.unsqueeze(1).expand(-1, expansion_factor, -1, -1)
            tau_embeddings = tau_embeddings.flatten(0, 1)

        # Reshape into (batch_size, 1, embedding_dim).
        state_embeddings = states.view(
            states.shape[0], 1, self.embedding_dim)

        # Calculate embeddings of states and taus.
        try:
            embeddings = (state_embeddings * tau_embeddings).view(
                states.shape[0] * N, *states.shape[1:])
        except:
            import ipdb; ipdb.set_trace()

        return embeddings



