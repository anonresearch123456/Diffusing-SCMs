import torch
import torch.nn as nn
from collections import OrderedDict


class BasicBlock(nn.Module):
    """
    A simple residual block for 3D data without batchnorm or dropout.
    It applies two 3D convolutions with ReLU activations and adds a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        # If the input and output dimensions differ, use a 1x1 convolution to match them.
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=True
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ThreeDCNN(nn.Module):
    """
    A 3D CNN with a similar interface to your 3D ResNet but without batchnorm or dropout.
    
    Args:
        in_channels (int): Number of input channels.
        n_outputs (int): Number of output classes.
        n_blocks (int): Number of residual blocks (must be at least 2).
        n_basefilters (int): Number of filters in the first convolution.
        no_pooling (bool): If True, skips the initial max pooling.
    """
    def __init__(self, in_channels: int, n_outputs: int, n_blocks: int = 4,
                 n_basefilters: int = 32, no_pooling: bool = False):
        super(ThreeDCNN, self).__init__()
        
        # Initial convolution: using a 7x7x7 kernel, stride 2, and padding 3.
        self.conv1 = nn.Conv3d(
            in_channels,
            n_basefilters,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True  # Bias is enabled since we are not using batchnorm.
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2) if not no_pooling else nn.Identity()

        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, but got {n_blocks}")

        # Build a sequence of residual blocks.
        blocks = []
        # The first block keeps the same number of filters.
        blocks.append(("block1", BasicBlock(n_basefilters, n_basefilters, stride=1)))
        n_filters = n_basefilters
        # Each subsequent block doubles the number of filters and downsamples spatially.
        for i in range(n_blocks - 1):
            blocks.append(
                (f"block{i+2}", BasicBlock(n_filters, n_filters * 2, stride=2))
            )
            n_filters *= 2

        self.feature_size = n_filters
        self.blocks = nn.Sequential(OrderedDict(blocks))
        
        # Global average pooling to reduce the spatial dimensions to 1x1x1.
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(n_filters, n_outputs, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.blocks(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        return self.fc(out), out

# Example usage:
if __name__ == "__main__":
    # Create a random tensor with shape (batch_size, channels, depth, height, width)
    x = torch.randn(2, 1, 64, 64, 64)
    model = ThreeDCNN(in_channels=1, n_outputs=10, n_blocks=4, n_basefilters=32, no_pooling=False)
    output = model(x)
    print("Output shape:", output.shape)  # Expected: (2, 10)
