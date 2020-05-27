import torch
import torch.nn as nn
import numpy as np
import copy
from src.model_trainer import ValueNetwork, renormalize

try:
    from geffnet.gen_efficientnet import _gen_efficientnet_condconv, _gen_efficientnet, _create_model
    from geffnet.efficientnet_builder import decode_arch_def, round_channels, resolve_act_layer, resolve_bn_args
except:
    # Do nothing; will crash on its own if need be, and don't want to force
    # installation of geffnet for normal workflows.
    print("Failed to import effnet code")
    pass


def curried_groupnorm(*args):
    return nn.GroupNorm(1, *args)


def curried_identity(*args):
    return nn.Identity()


class RLEffNet(nn.Module):
    def __init__(self,
                 imagesize=100,
                 condconv=True,
                 norm_type="ln",
                 in_channels=4,
                 drop_connect_rate=0.1,
                 drop_rate=0.1,):

        super().__init__()

        if condconv:
            variant = 'efficientnet_cc_b0_4e'
            gen_function = _gen_efficientnet_condconv
        else:
            gen_function = _gen_efficientnet
            variant = "efficientnet_b0"

        if norm_type == "bn":
            norm = nn.BatchNorm2d
        elif norm_type == "in":
            norm = nn.InstanceNorm2d
        elif norm_type == "ln":
            norm = curried_groupnorm
        elif norm_type == "none":
            norm = curried_identity

        scaling_factor = np.log(imagesize/224.)/np.log(1.15)
        width_scaling_factor = 1.2**scaling_factor
        depth_scaling_factor = 1.1**scaling_factor
        model = gen_function(variant,
                             channel_multiplier=width_scaling_factor,
                             depth_multiplier=depth_scaling_factor,
                             pretrained=False,
                             in_chans=in_channels,
                             norm_layer=norm,
                             drop_connect_rate=drop_connect_rate,
                             drop_rate=drop_rate)

        self.network = nn.Sequential(model.conv_stem,
                                     model.bn1,
                                     model.act1,
                                     model.blocks)

        fake_input = torch.zeros(1, in_channels, imagesize, imagesize)
        fake_output = self.network(fake_input)
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]

    def forward(self, x):
        return self.network(x)


class EffnetTransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 action_dim=32,
                 args=None,
                 blocks=16,
                 pixels=36,
                 limit=300,
                 norm_type="bn",
                 renormalize=True,
                 drop_connect_rate=0.,
                 hidden_size=-1,
                 drop_rate=0.,):
        super().__init__()
        self.hidden_size = channels
        self.action_dim = action_dim
        self.args = args
        self.renormalize = renormalize
        self.action_embedding = nn.Embedding(num_actions, action_dim)

        if norm_type == "bn":
            norm = nn.BatchNorm2d
        elif norm_type == "in":
            norm = nn.InstanceNorm2d
        elif norm_type == "ln":
            norm = curried_groupnorm
        elif norm_type == "none":
            norm = curried_identity

        self.model = generate_condconv_resnet(channels+action_dim,
                                              blocks,
                                              norm_layer=norm,
                                              drop_connect_rate=drop_connect_rate,
                                              drop_rate=drop_rate)

        self.reward_predictor = ValueNetwork(channels,
                                             pixels=pixels,
                                             limit=limit,
                                             norm_type="bn" if
                                             norm_type == "bn" else "none")
        self.train()

    def forward(self, x, action):
        action_emb = self.action_embedding(action)
        action_emb = action_emb[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        stacked_image = torch.cat([x, action_emb], 1)
        next_state = self.model(stacked_image)
        next_state = next_state[:, :-self.action_dim]
        if self.renormalize:
            next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward


def generate_condconv_resnet(hidden_size,
                             blocks,
                             experts_multiplier=1,
                             **kwargs):
    """Creates an efficientnet-condconv model."""
    arch_def = [
      ['ir_r{}_k3_s1_e4_c{}_se0.25_cc{}'.format(blocks, hidden_size, 4*experts_multiplier)],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, 1., experts_multiplier=experts_multiplier),
        num_features=round_channels(1280, 1., 8, None),
        stem_size=hidden_size,
        channel_multiplier=1.,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    model = _create_model(model_kwargs, 'efficientnet_cc_b0_4e', False)
    resnet = copy.deepcopy(model.blocks)  # only part we want
    del model  # Scratch the rest for memory
    return resnet