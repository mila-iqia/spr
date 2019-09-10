import torch
import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class ImpalaCNN(nn.Module):
    def __init__(self, input_channels, args):
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

    def forward(self, inputs):
        out = inputs
        if self.downsample:
            out = self.layer3(self.layer2(self.layer1(out)))
        else:
            out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return F.relu(self.final_linear(self.flatten(out)))


class NatureCNN(nn.Module):
    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        self.hidden_size = self.feature_size
        self.input_channels = input_channels
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        self.final_conv_size = 64 * 3 * 3
        self.final_conv_shape = None

        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=5, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(self.final_conv_size, args.feature_size)
        )

        self.train()

    def forward(self, inputs, fmaps=False):
        f5 = self.main[:4](inputs)
        out = self.main[4:](f5)
        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'out': out
            }
        return out
