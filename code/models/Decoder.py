import torch
import torch.nn as nn
from Residual import ResidualBlock

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = self._build()
    
    def _build(self):
        pass

    def forward(self, x):
        x = self.layers(x)
        return x