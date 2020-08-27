import torch
import torch.nn as nn
from Residual import ResidualBlock

# Author's original implementation: https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
# Other references:
# https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
# https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py

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
            num_resblocks: the number of residual blocks used in the decoder
            res_channels: the number of channels that are used by the residual blocks
            out_channels: the number of channels that are used by the output layer, should be the same to the number of the input image channels
        """
        super(Decoder, self).__init__()
        self.layers = self._build(in_channels, hidden_channels, num_resblocks, res_channels, out_channels)
    
    def _build(self, in_channels, hidden_channels, num_resblocks, res_channels, out_channels):
        layers = [
            # here, we want to use the same padding to keep the output size the same as the input size
            # padding = (s(n - 1) - n + f) / 2
            # ((n - 1) -n + 3) / 2 = 1
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
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
        layers += [
            # since the encoder halves the input image size twice with the first two layers,
            # the decoder will double the size using twice with the following layers
            # p = (2(in - 1) + 4 - 2in) / 2 = 1
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # p = (2(in - 1) + 4 - 2in) / 2 = 1
            nn.ConvTranspose2d(in_channels=hidden_channels//2, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == "__main__":
    net = Decoder(128, 128, 2, 32, 3)
    print(net)

    sample = torch.randn((1, 128, 32, 32))
    output = net(sample)
    print(output.shape)