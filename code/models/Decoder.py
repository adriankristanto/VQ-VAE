import torch
import torch.nn as nn
from Residual import ResidualBlock

# reference: https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb

class Decoder(nn.Module):

    """
    According to the paper, the decoder is composed of two residual 3 × 3 blocks, 
    followed by two transposed convolutions with stride 2 and kernel size (4, 4)
    Similar to the encoder, the author added a conv layer before the residual blocks
    with kernel_size (3, 3) and stride = 1 and padding = 1
    """

    def __init__(self, in_channels, hidden_channels, num_resblocks, res_channels, out_channels):
        """
            in_channels: the number of channels that are used by the input layer
            hidden_channels: the number of channels that are used by the hidden conv layers
            num_resblocks: the number of residual blocks used in the encoder
            res_channels: the number of channels that are used by the residual blocks
            out_channels: the number of channels that are used by the output layer, should be the same to the number of the input image channels
        """
        super(Decoder, self).__init__()
        self.layers = self._build(in_channels, hidden_channels, num_resblocks, res_channels, out_channels)
    
    def _build(self, in_channels, hidden_channels, num_resblocks, res_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        ]
        layers += [
            Residual(hidden_channels, res_channels, hidden_channels) for _ in range(num_resblocks)
        ]
        layers += [
            nn.ReLU()
        ]
        layers += [
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=hidden_channels//2, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x