import torch
import torch.nn as nn
import numpy as np

try:
    from geffnet.gen_efficientnet import _gen_efficientnet_condconv, _gen_efficientnet
except:
    # Do nothing; will crash on its own if need be, and don't want to force
    # installation of geffnet for normal workflows.
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

