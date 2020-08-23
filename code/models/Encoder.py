import torch 
import torch.nn as nn
from Residual import ResidualBlock

class Encoder(nn.Module):

    """
    According to the paper, the encoder is composed of 2 strided convolutional layers 
    with stride 2 and kernel size (4, 4), followed by two residual 3 × 3 blocks.
    However, the author added a Conv layer in between the strided convolutional layers and the resblock
    with kernel size = (3, 3) and stride = 1.
    reference: https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
    """

    def __init__(self, in_channels, hidden_channels, num_resblocks, res_channels):
        super(Encoder, self).__init__()
        self.layers = self._build(in_channels, hidden_channels, num_resblocks, res_channels)
    
    def _build(self, in_channels, hidden_channels, num_resblocks, res_channels):
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels // 2, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ]
        layers += [
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ]
        layers += [
            ResidualBlock(hidden_channels, res_channels, hidden_channels) for _ in range(num_resblocks)
        ]
        layers += [
            nn.ReLU()
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == "__main__":
    net = Encoder(3, 128, 2, 32)
    print(net)