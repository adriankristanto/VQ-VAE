import torch
import torch.nn as nn
import torch.nn.functional as F 
from Encoder import Encoder
from VectorQuantizer import VectorQuantizer
from Decoder import Decoder

class VQVAE(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_resblocks, res_channels, D, K, beta, gamma):
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