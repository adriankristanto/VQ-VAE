import torch
import torch.nn as nn 
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    """
    According to the VQ-VAE paper, both the encoder and decoder have residual 3x3 blocks, 
    which are implemented as ReLU, 3x3 Conv, ReLU, 1x1 Conv
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.resblock = self._build(channels)
    
    def _build(self, channels):
        resblock = nn.Sequential(
            nn.ReLU(),
            # the conv layer should use the same padding, thus,
            # NOTE: in == out == n
            # padding = (s(n - 1) - n + f) / 2
            # ((n - 1) -n + 3) / 2 = 1
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            # padding = (s(out - 1) - n + f) / 2
            # ((n - 1) -n + 1) / 2 = 0
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=(1, 1), stride=1, padding=0, bias=False)
        )
        return resblock

    def forward(self, x):
        res_x = self.resblock(x)
        x += res_x
        return x