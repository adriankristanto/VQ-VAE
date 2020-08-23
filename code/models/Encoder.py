import torch 
import torch.nn as nn
from Residual import ResidualBlock

# reference: https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb

class Encoder(nn.Module):

    """
    According to the paper, the encoder is composed of 2 strided convolutional layers 
    with stride 2 and kernel size (4, 4), followed by two residual 3 × 3 blocks.
    However, the author added a Conv layer in between the strided convolutional layers and the resblock
    with kernel size = (3, 3) and stride = 1.
    """

    def __init__(self, in_channels, hidden_channels, num_resblocks, res_channels):
        """
            in_channels: the number of channels that the input image has
            hidden_channels: the number of channels that are used by the hidden conv layers
            num_resblocks: the number of residual blocks used in the encoder
            res_channels: the number of channels that are used by the residual blocks
        """
        super(Encoder, self).__init__()
        self.layers = self._build(in_channels, hidden_channels, num_resblocks, res_channels)
    
    def _build(self, in_channels, hidden_channels, num_resblocks, res_channels):
        # NOTE: the out_channels follows the author's implementation
        layers = [
            # here, I assume that we want the size of the output to be half of the input size
            # padding = (2(n/2 - 1) -n + 4) / 2 = 1
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # padding = (2(n/2 - 1) -n + 4) / 2 = 1
            nn.Conv2d(in_channels=hidden_channels // 2, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ]
        layers += [
            # here, we want to use the same padding to keep the output size the same as the input size
            # padding = (s(n - 1) - n + f) / 2
            # ((n - 1) -n + 3) / 2 = 1
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ]
        layers += [
            # here, we create num_resblocks number of residual blocks
            ResidualBlock(hidden_channels, res_channels, hidden_channels) for _ in range(num_resblocks)
        ]
        layers += [
            # each resblock output is not wrapped by the activation function as the next block has ReLU as its first layer
            # however, the final resblock doesn't have anything to wrap its output with an activation function
            # therefore, here we wrap the output of the final resblock with ReLU
            nn.ReLU()
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == "__main__":
    net = Encoder(3, 128, 2, 32)
    print(net)

    sample = torch.randn((1, 3, 128, 128))
    output = net(sample)
    print(output.shape)