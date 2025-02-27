# This file is part of Prototypical Additive Neural Network for Interpretable Classification (PANIC).
#
# PANIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PANIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PANIC. If not, see <https://www.gnu.org/licenses/>.
from collections import OrderedDict
from torch import nn


def conv3d(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
) -> nn.Module:
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False,
    )


class ConvBnReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.05,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_momentum: float = 0.05, stride: int = 1, dropout_p=0.2):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.dropout1 = nn.Dropout(p=dropout_p, inplace=True)

        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ThreeDResNet(nn.Module):
    def __init__(self, in_channels: int, n_outputs: int, n_blocks: int = 4, bn_momentum: float = 0.05, n_basefilters: int = 32, dropout_p: float=0.2,
                 no_pooling: bool = False):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            n_basefilters,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,)
        self.bn1 = nn.BatchNorm3d(n_basefilters, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(3, stride=2) if not no_pooling else nn.Identity()

        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, but got {n_blocks}")

        blocks = [
            ("block1", ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum, dropout_p=dropout_p))
        ]
        n_filters = n_basefilters
        for i in range(n_blocks - 1):
            blocks.append(
                (f"block{i+2}", ResBlock(n_filters, 2 * n_filters, bn_momentum=bn_momentum, stride=2, dropout_p=dropout_p))
            )
            n_filters *= 2

        self.feature_size = n_filters
        self.blocks = nn.Sequential(OrderedDict(blocks))
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(n_filters, n_outputs, bias=True)

#    def get_out_features(self):
#
#        return self.out_features

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.blocks(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        return self.fc(out), out


class ThreeDResNetFixedChannels(nn.Module):
    """
    3D ResNet variant with fixed number of channels across layers.
    Suitable for structured latent spaces with high-level features.
    """
    def __init__(self, 
                 in_channels: int, 
                 n_outputs: int, 
                 n_blocks: int = 4, 
                 bn_momentum: float = 0.05, 
                 n_basefilters: int = 32, 
                 dropout_p: float = 0.2,
                 no_pooling: bool = False):
        super().__init__()

        # Initial Convolution Layer
        self.conv1 = nn.Conv3d(
            in_channels,
            n_basefilters,
            kernel_size=3,
            stride=1,    # No initial striding
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(n_basefilters, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) if not no_pooling else nn.Identity()

        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, but got {n_blocks}")

        # Create ResBlocks with fixed channels
        blocks = []
        current_filters = n_basefilters
        for i in range(n_blocks):
            blocks.append(
                (f"block{i+1}", ResBlock(current_filters, current_filters, bn_momentum=bn_momentum, stride=1, dropout_p=dropout_p))
            )

        self.blocks = nn.Sequential(OrderedDict(blocks))
        self.feature_size = current_filters

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(current_filters, n_outputs, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.blocks(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return logits, out
    

class ThreeDResNetEncoded(nn.Module):
    """
    Adapted 3D ResNet tailored for encoded 3D MRI data with small spatial dimensions.
    Avoids aggressive striding and excessive pooling to preserve spatial information.
    """
    def __init__(self, 
                 in_channels: int, 
                 n_outputs: int, 
                 n_blocks: int = 4, 
                 bn_momentum: float = 0.05, 
                 n_basefilters: int = 32, 
                 dropout_p: float = 0.2,
                 no_pooling: bool = False):
        """
        Initializes the ThreeDResNetEncoded model.

        Parameters:
        - in_channels (int): Number of input channels.
        - n_outputs (int): Number of output classes.
        - n_blocks (int): Total number of ResBlocks.
        - bn_momentum (float): Momentum for BatchNorm layers.
        - n_basefilters (int): Number of base filters for the initial convolution.
        - dropout_p (float): Dropout probability in ResBlocks.
        - use_pooling (bool): Whether to use initial max pooling.
        """
        super().__init__()

        # Initial Convolution Layer
        # Using smaller kernel_size and stride to accommodate small spatial dimensions
        self.conv1 = nn.Conv3d(
            in_channels,
            n_basefilters,
            kernel_size=3,
            stride=1,    # Changed from 2 to 1
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(n_basefilters, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=True)
        # Avoid aggressive pooling; use pooling only if specified
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) if not no_pooling else nn.Identity()

        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, but got {n_blocks}")

        # Dynamically create ResBlocks
        blocks = []
        current_filters = n_basefilters
        for i in range(n_blocks):
            if i == 0:
                # First block: no downsampling
                blocks.append(
                    (f"block{i+1}", ResBlock(current_filters, current_filters, bn_momentum=bn_momentum, stride=1, dropout_p=dropout_p))
                )
            else:
                # Subsequent blocks: double the filters and apply stride=2 for downsampling
                out_filters = current_filters * 2
                blocks.append(
                    (f"block{i+1}", ResBlock(current_filters, out_filters, bn_momentum=bn_momentum, stride=2, dropout_p=dropout_p))
                )
                current_filters = out_filters

        self.blocks = nn.Sequential(OrderedDict(blocks))

        self.feature_size = current_filters

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(current_filters, n_outputs, bias=True)

    def forward(self, x):
        """
        Forward pass of the ThreeDResNetEncoded model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)

        Returns:
        - torch.Tensor: Output logits of shape (batch_size, n_outputs)
        - torch.Tensor: Feature vector after global average pooling
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.blocks(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return logits, out
