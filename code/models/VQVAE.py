import torch
import torch.nn as nn
import torch.nn.functional as F 
from Encoder import Encoder
from VectorQuantizer import VectorQuantizerEMA
from Decoder import Decoder

class VQVAE(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_resblocks, res_channels, D, K, beta=0.25, gamma=0.99):
        """
        in_channels: number of channels of the input image
        hidden_channels: the number of channels that are used by the hidden conv layers
        num_resblocks: the number of residual blocks used in both the encoder and the decoder
        res_channels: the number of channels that are used by the residual blocks
        D: dimensionality of each embedding vector, or embedding_dim in the sonnet's implementation
        K: the size of the discrete space (the number of embedding vectors), or num_embeddings in the sonnet's implementation
        beta: the hyperparameter that acts as a weighting to the lost term, or commitment_cost in the sonnet's implementation
            recommendation from the paper, beta=0.25
        gamma: controls the speed of the EMA, or decay in the sonnet's implementation
            recommendation from the paper, gamma=0.99
        """
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, num_resblocks, res_channels)
        # the following is the additional layer added in the author's original implementation
        # to make sure that the number of channels equals to D (embedding dimension)
        self.pre_vq = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=D,
            kernel_size=1,
            stride=1
        )
        self.vectorquantizer = VectorQuantizerEMA(D, K, beta, gamma)
        self.decoder = Decoder(D, hidden_channels, num_resblocks, res_channels, in_channels)
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.pre_vq(x)
        quantized, loss, perplexity, _, _, _ = self.vectorquantizer(x)
        return quantized, loss, perplexity
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        quantized, loss, perplexity = self.encode(x)
        x = self.decode(quantized)
        return x, loss, perplexity, quantized

if __name__ == "__main__":
    net = VQVAE(3, 128, 2, 32, 64, 512)
    x = torch.randn((1, 3, 128, 128))
    outputs = net(x)
    print(outputs[0].shape)
    print(outputs[3].shape)