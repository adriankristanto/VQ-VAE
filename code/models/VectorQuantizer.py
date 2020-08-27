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

    def __init__(self, D, K, beta=0.25, gamma=0.99, epsilon=1e-5):
        """
        According to the paper,
            D: dimensionality of each embedding vector, or embedding_dim in the sonnet's implementation
            K: the size of the discrete space (the number of embedding vectors), or num_embeddings in the sonnet's implementation
            beta: the hyperparameter that acts as a weighting to the lost term, or commitment_cost in the sonnet's implementation
                recommendation from the paper, beta=0.25
            gamma: controls the speed of the EMA, or decay in the sonnet's implementation
                recommendation from the paper, gamma=0.99
            epsilon: to avoid numerical instability (such as division by zero)
                from the original implementation, epsilon=1e-5
        """
        super(VectorQuantizerEMA, self).__init__()
        # assign the parameters to self
        self.D = D
        self.K = K
        # in this implementation, the loss will be multiplied by beta during the forward pass of the VQ layer
        # instead of during training
        # so that, during training, we can simply add up the latent loss and the reconstruction loss
        # without having to multiply to latent loss with beta

        # reference: https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
        # the above implementation return the latent loss without multiplying it with beta
        # it only multiplies the latent loss with beta during training
        # the original sonnet implementation, however, multiplied the latent loss with beta during the forward pass
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, x):
        # firstly, flatten all other dimension except for the last one
        # for example, in the sonnet documentation, input tensor of shape (16, 32, 32, 64)
        # will be reshaped to (16 * 32 * 32, 64)
        # which means we have 16 * 32 * 32 tensors of 64 dimensions
        # 64 here is the input parameter D in our implementation
        x = x.view(-1, self.D)
        return x

if __name__ == "__main__":
    tensor = torch.randn((16, 32, 32, 64))
    net = VectorQuantizerEMA(tensor.shape[3], tensor.shape[0] * tensor.shape[1] * tensor.shape[2])
    print(net(tensor).shape)