import torch
import torch.nn as nn 

# Sonnet's implementation: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
# Other references:
# https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
# https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py

class VectorQuantizerEMA(nn.Module):

    """
    VectorQuantizer with Exponentially Moving Averages (EMA)

    In the paper, Exponentially Moving Averages (EMA) mentioned as an 
    alternative to the loss function to update the embedding vectors.
    
    Advantages:
        1. The EMA update method is independent of the optimiser used to train the model.
        2. Additionally, using this method results in faster training.
    """

    def __init__(self, D, K, beta, gamma, epsilon):
        """
        According to the paper,
            D: dimensionality of each embedding vector, or embedding_dim in the sonnet's implementation
            K: the size of the discrete space (the number of embedding vectors), or num_embeddings in the sonnet's implementation
            beta: the hyperparameter that acts as a weighting to the lost term, or commitment_cost in the sonnet's implementation
            gamma: controls the speed of the EMA, or decay in the sonnet's implementation
            epsilon: to avoid numerical instability (such as division by zero)
        """
        super(VectorQuantizerEMA, self).__init__()
        # assign the parameters to self
        self.D = D
        self.K = K
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        