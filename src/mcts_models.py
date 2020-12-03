import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.models.mlp import MlpModel
from src.model_trainer import from_categorical, NetworkOutput
from src.rlpyt_models import Conv2dModel

from rlpyt.models.utils import scale_grad, update_state_dict
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from src.utils import count_parameters, dummy_context_mgr, Intensity
import numpy as np
from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    CenterCrop, \
    RandomResizedCrop
from kornia.filters import GaussianBlur2d
import copy
import wandb


class MCTSModel(torch.nn.Module):
    def __init__(self, image_shape, output_size, jumps=0, spr=0, imagesize=84,
                 augmentation=['none'], target_augmentation=0, eval_augmentation=0,
                 dynamics_blocks=0,
                 norm_type='bn',
                 dqn_hidden_size=256,
                 model_rl=0,
                 momentum_tau=0.01,
                 renormalize=1,
                 dropout=0.,
                 residual_tm=0.,
                 projection='q_l1',
                 predictor='linear',
                 q_l1_type='',
                 distributional=0,
                 aug_prob=0.8):
        super().__init__()
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

        self.transforms = []
        self.eval_transforms = []
        self.distributional = distributional
        self.uses_augmentation = False
        self.aug_prob = aug_prob
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop((84, 84))
                # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
                eval_transformation = CenterCrop((84, 84))
                self.uses_augmentation = True
                imagesize = 84
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        fake_input = torch.zeros(1, f*c, imagesize, imagesize)
        fake_output = self.conv(fake_input)
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        self.jumps = jumps
        self.model_rl = model_rl
        self.use_spr = spr
        self.target_augmentation = target_augmentation
        self.eval_augmentation = eval_augmentation
        self.num_actions = output_size

        self.head = ValueHead(input_channels=self.hidden_size,
                              pixels=self.pixels,
                              output_size=1)

        self.policy_head = PolicyHead(input_channels=self.hidden_size,
                                      output_size=self.num_actions,
                                      pixels=self.pixels)

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

        if self.use_spr:
            self.momentum_tau = momentum_tau

            if projection == "mlp":
                self.projection = nn.Sequential(
                                            nn.Flatten(-3, -1),
                                            nn.Linear(self.pixels*self.hidden_size, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(),
                                            nn.Linear(512, 256)
                                            )
                self.target_projection = self.projection
                global_spr_size = 256
            elif projection == "q_l1":
                self.projection = QL1Head([self.policy_head.network[:2],
                                           self.head.network[:2]],
                                          noisy="noisy" in q_l1_type,
                                          relu="relu" in q_l1_type)
                global_spr_size = 2*dqn_hidden_size
                self.target_projection = self.projection
            if predictor == "mlp":
                self.predictor = nn.Sequential(
                    nn.Linear(global_spr_size, global_spr_size*2),
                    nn.BatchNorm1d(global_spr_size*2),
                    nn.ReLU(),
                    nn.Linear(global_spr_size*2, global_spr_size)
                )
            elif predictor == "linear":
                self.predictor = nn.Sequential(
                    nn.Linear(global_spr_size, global_spr_size),
                )
            elif predictor == "none":
                self.predictor = nn.Identity()

            self.target_encoder = copy.deepcopy(self.conv)
            self.target_projection = copy.deepcopy(self.target_projection)
            for param in (list(self.target_encoder.parameters()) +
                          list(self.target_projection.parameters())):
                param.requires_grad = False

        print("Initialized model with {} parameters".format(count_parameters(self)))
        self.renormalize = renormalize

    def set_sampling(self, sampling):
        if self.noisy:
            self.head.set_sampling(sampling)

    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def do_spr_loss(self, pred_latents, targets, observation):
        pred_latents = self.projection(pred_latents)
        pred_latents = self.predictor(pred_latents)

        targets = targets.view(-1, observation.shape[1],
                                             self.jumps+1, targets.shape[-1]).transpose(1, 2)
        latents = pred_latents.view(-1, observation.shape[1],
                                             self.jumps+1, pred_latents.shape[-1]).transpose(1, 2)

        spr_loss = self.spr_loss(latents, targets).view(-1, observation.shape[1]) # split to batch, jumps

        update_state_dict(self.target_encoder,
                          self.conv.state_dict(),
                          self.momentum_tau)
        update_state_dict(self.target_projection,
                          self.projection.state_dict(),
                          self.momentum_tau)
        return spr_loss

    def stem_forward(self, img, prev_action=None, prev_reward=None):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        if self.renormalize:
            conv_out = renormalize(conv_out, -3)
        return conv_out

    def head_forward(self,
                     conv_out,
                     prev_action,
                     prev_reward,
                     logits=False):
        lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)
        p = self.head(conv_out)

        if self.distributional:
            if logits:
                p = F.log_softmax(p, dim=-1)
            else:
                p = F.softmax(p, dim=-1)
        else:
            p = p.squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    def forward(self, observation,
                prev_action, prev_reward,
                train=False, eval=False):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity.

        For convenience reasons with DistributedDataParallel the forward method
        has been split into two cases, one for training and one for eval.
        """
        if eval:
            self.eval()
        else:
            self.train()
        if train:
            log_pred_ps = []
            pred_reward = []
            pred_latents = []
            pred_policy_logits = []
            input_obs = observation[0].flatten(1, 2)
            input_obs = self.transform(input_obs, augment=True)
            latent = self.stem_forward(input_obs,
                                       prev_action[0],
                                       prev_reward[0])
            log_pred_ps.append(self.head_forward(latent,
                                                 prev_action[0],
                                                 prev_reward[0],
                                                 logits=True))
            pred_latents.append(latent)
            if self.jumps > 0:
                pred_rew = self.dynamics_model.reward_predictor(pred_latents[0])
                policy = self.policy_head(pred_latents[0])
                pred_policy_logits.append(F.log_softmax(policy, -1))
                pred_reward.append(F.log_softmax(pred_rew, -1))

                for j in range(1, self.jumps + 1):
                    latent, pred_rew, policy, value = self.step(latent, prev_action[j])
                    pred_rew = pred_rew[:observation.shape[1]]
                    pred_latents.append(latent)
                    pred_reward.append(F.log_softmax(pred_rew, -1))
                    pred_policy_logits.append(F.log_softmax(policy, -1))

            if self.model_rl > 0:
                for i in range(1, len(pred_latents)):
                    log_pred_ps.append(self.head_forward(pred_latents[i],
                                                         prev_action[i],
                                                         prev_reward[i],
                                                         logits=True))

            with torch.no_grad():
                target_images = observation[:self.jumps+1].transpose(0, 1).flatten(2, 3)
                target_latents = self.stem_forward(self.transform(target_images, augment=False))

            pred_latents = torch.stack(pred_latents, 1)
            if self.use_spr:
                spr_loss = self.do_spr_loss(pred_latents.flatten(0, 1),
                                            target_latents.flatten(0, 1),
                                            observation)
            else:
                spr_loss = torch.zeros((self.jumps + 1, observation.shape[1]), device=latent.device)

            return log_pred_ps,\
                   pred_reward,\
                   spr_loss, \
                   pred_policy_logits

        else:
            aug_factor = self.target_augmentation if not eval else self.eval_augmentation
            observation = observation.flatten(-4, -3)
            stacked_observation = observation.unsqueeze(1).repeat(1, max(1, aug_factor), 1, 1, 1)
            stacked_observation = stacked_observation.view(-1, *observation.shape[1:])

            img = self.transform(stacked_observation, aug_factor)

            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

            conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
            if self.renormalize:
                conv_out = renormalize(conv_out, -3)
            p = self.head(conv_out)

            if self.distributional:
                p = F.softmax(p, dim=-1)
            else:
                p = p.squeeze(-1)

            p = p.view(observation.shape[0],
                       max(1, aug_factor),
                       *p.shape[1:]).mean(1)

            # Restore leading dimensions: [T,B], [B], or [], as input.
            p = restore_leading_dims(p, lead_dim, T, B)

            return p

    def initial_inference(self, obs, actions=None, logits=False):
        if len(obs.shape) == 5:
            obs = obs.flatten(1, 2)
        obs = self.transform(obs, self.eval_augmentation)
        hidden_state = self.conv(obs)
        policy_logits = self.policy_head(hidden_state)
        value_logits = self.head(hidden_state)
        reward_logits = self.dynamics_model.reward_predictor(hidden_state)

        if logits:
            return NetworkOutput(hidden_state, reward_logits, policy_logits, value_logits)

        value = from_categorical(value_logits, logits=True, limit=10)  # TODO Make these configurable
        reward = from_categorical(reward_logits, logits=True, limit=1)
        return NetworkOutput(hidden_state, reward, policy_logits, value)

    def inference(self, state, action):
        next_state, reward_logits, \
        policy_logits, value_logits = self.step(state, action)
        value = from_categorical(value_logits, logits=True, limit=10)  # TODO Make these configurable
        reward = from_categorical(reward_logits, logits=True, limit=1)

        return NetworkOutput(next_state, reward, policy_logits, value)

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        policy_logits = self.policy_head(next_state)
        value_logits = self.head(next_state)
        return next_state, reward_logits, policy_logits, value_logits

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = maybe_transform(image, transform,
                                        eval_transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float()/255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None,
                                                     flat_images)
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images


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
        self.reward_predictor = RewardHead(channels, output_size=1,
                                           pixels=pixels,
                                           norm_type=norm_type)
        self.train()

    def forward(self, x, action):
        if action.dim() < 1:
            action = action.unsqueeze(0)
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


class QL1Head(nn.Module):
    def __init__(self,
                 heads,
                 noisy=False,
                 relu=False):
        """
        :param heads: Encoder heads, wrapped in nn.Sequential
        :param type:
        """
        super().__init__()
        self.encoders = nn.ModuleList(*heads)
        self.relu = relu
        self.noisy = noisy

    def forward(self, x):
        representations = []
        for encoder in self.encoders:
            [setattr(module, "noise_override", self.noisy) for module in encoder]
            encoder.noise_override = self.noisy
            representations.append(encoder(x))
            [setattr(module, "noise_override", None) for module in encoder]
        representation = torch.cat(representations, -1)
        if self.relu:
            representation = F.relu(representation)

        return representation


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
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

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

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
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
        if self.noise_override is None:
            use_noise = self.training or self.sampling
        else:
            use_noise = self.noise_override
        if use_noise:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


def maybe_transform(image, transform, alt_transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        base_images = alt_transform(image)
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * base_images
        return processed_images


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


class PolicyHead(nn.Module):
    def __init__(self,
                 input_channels,
                 output_size,
                 filters=1,
                 pixels=36,
                 norm_type="bn"):
        super().__init__()
        layers = [nn.Conv2d(input_channels, filters, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(filters, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*filters, 256),
                  nn.ReLU(),
                  nn.Linear(256, output_size)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


class RewardHead(nn.Module):
    def __init__(self,
                 input_channels,
                 output_size=1,
                 filters=1,
                 pixels=36,
                 norm_type="bn"):
        super().__init__()
        layers = [nn.Conv2d(input_channels, filters, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(filters, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*filters, 256),
                  nn.ReLU(),
                  nn.Linear(256, output_size)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


class ValueHead(nn.Module):
    def __init__(self,
                 input_channels,
                 output_size=1,
                 filters=1,
                 pixels=36,
                 norm_type="bn"):
        super().__init__()
        layers = [nn.Conv2d(input_channels, filters, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(filters, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*filters, 256),
                  nn.ReLU(),
                  nn.Linear(256, output_size)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)