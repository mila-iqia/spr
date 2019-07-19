# Dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    r"""Implements a sequentiable module in torch for flattening features."""
    
    def forward(self, x):
        r"""Implements the forward pass.

        inputs:
        -------
        x: Torch tensor to be flattened. SHAPE: [<batch_size>, ...]

        outputs:
        --------
        x_flatten (implicit): The flattened tensor for input x. 
            SHAPE: [<batch_size>, <feat_size>]
        """
        return x.view(x.size(0), -1)


class Conv2dSame(torch.nn.Module):
    r"""Implements a sequentiable module in torch for same-shaped 2D convs."""

    def __init__(self, in_channels, out_channels, kernel_size, 
            bias=True, padding_layer=nn.ReflectionPad2d):
        r"""The constructor.

        inputs:
        -------
        in_channels: The number of input channels in the features.

        out_channels: The number of output channels in the features.

        kernel_size: The size of the square kernel.

        bias=True: Whether to include bias in the convolution layer.

        padding_layer=nn.ReflectionPad2d: Padding layer for same-shaped
            convolutional layer output.

        outputs:
        --------
        """
        super().__init__()
        # Calculate the appropriate padding for same-shaped output.
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        # Add the padding layer followed by the 2D conv layer.
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        r"""Implements the forward pass.

        inputs:
        -------
        x: The torch tensor. SHAPE: [<batch_size>, <in_channels>, 
                                                    <width>, <height>].

        outputs:
        --------
        x_conv (implicit): The output of the convolution layer.
        """
        return self.net(x)


class ResidualBlock(nn.Module):
    r"""Implements a sequentiable torch module for residual block."""

    def __init__(self, in_channels, out_channels):
        r"""The constructor.

        inputs:
        -------
        in_channels: The number of channels in the input.

        out_channels: The number of channels in the output.

        outputs:
        --------
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3)
        )

    def forward(self, x):
        r"""Implments the forward pass.

        inputs:
        -------
        x: The input torch tensor. SHAPE: [<batch_size>, <in_channel>,
                                        <width>, <height>]

        outputs:
        --------
        x_residual (implicit): The forward pass through the block. 
        """
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class ImpalaCNN(nn.Module):
    r"""Implements ImpalaCNN architecture."""

    def __init__(self, input_channels, args):
        r"""The constructor.

        inputs:
        -------
        input_channels: The number of channels in the input feature.

        args: The configs for the architecture.

        outputs:
        --------
        """
        super(ImpalaCNN, self).__init__()
        self.hidden_size = args.feature_size
        self.depths = [16, 32, 32, 32]
        self.downsample = not args.no_downsample
        self.layer1 = self._make_layer(input_channels, self.depths[0])
        self.layer2 = self._make_layer(self.depths[0], self.depths[1])
        self.layer3 = self._make_layer(self.depths[1], self.depths[2])
        self.layer4 = self._make_layer(self.depths[2], self.depths[3])
        if self.downsample:
            self.final_conv_size = 32 * 9 * 9
        else:
            self.final_conv_size = 32 * 12 * 9
        self.final_linear = nn.Linear(self.final_conv_size, self.hidden_size)
        self.flatten = Flatten()
        # Set the encoder model to train mode.
        self.train()

    def _make_layer(self, in_channels, depth):
        r"""Creates a residual architecture based layer block.

        inputs:
        -------
        in_channels: The number of channels in the feature.

        depth: The number of output channels in the residual and conv blocks.

        outputs:
        --------
        block_output (implicit): Block output. SHAPE: [<batch_size>, 
                <channel=depth>, <width>, <height>]
        """
        return nn.Sequential(
            Conv2dSame(in_channels, depth, 3),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth)
        )

    def forward(self, inputs):
        r"""Implements forward pass of the block.

        inputs:
        -------
        inputs: SHAPE: [<batch_size>, <in_channel>, <width>, <height>].

        outputs:
        --------
        block_output: SHAPE: [<batch_size>, <out_channel>, <width>, <height>].
        """
        out = inputs
        if self.downsample:
            out = self.layer3(self.layer2(self.layer1(out)))
        else:
            out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return F.relu(self.final_linear(self.flatten(out)))


class NatureCNN(nn.Module):
    r"""Implements NatureCNN architecture."""

    def __init__(self, input_channels, args):
         r"""The constructor.

        inputs:
        -------
        input_channels: The number of channels in the input feature.

        args: The configs for the architecture.

        outputs:
        --------
        """
        super().__init__()
        self.feature_size = args.feature_size
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = input_channels
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()
        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        else:
            self.final_conv_size = 64 * 9 * 6
            self.final_conv_shape = (64, 9, 6)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 128, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(128, 64, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        # Set mode of the encoder model to train.
        self.train()

    def forward(self, inputs, fmaps=False):
        r"""Implements the forward pass of the architecture.

        inputs:
        -------
        inputs: SHAPE: [<batch_size>, <in_channels>, <width>, <height>].

        fmaps=False: Whether to return all feature maps in the forward pass.

        outputs:
        --------
        out: Output. If fmaps, dictionary of feature maps. Otherwise, the 
            output of the encoder model.
        """
        f5 = self.main[:6](inputs)
        f7 = self.main[6:8](f5)
        out = self.main[8:](f7)
        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'f7': f7.permute(0, 2, 3, 1),
                'out': out
            }
        return out