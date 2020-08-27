import torch
import torch.nn as nn 
import torch.nn.functional as F

# Sonnet's implementation: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
# Other references:
# https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
# https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
# NOTE: this implementation follows https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py closely

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

        # create the embedding layer's weights
        # initialise the weights using randn
        embedding_init = torch.randn(D, K)
        # why use register_buffer? because we want to save the embedding weights, etc. into saved state
        # for training continuation
        # why not use nn.Embedding or register_parameter? here, we update the weights using EMA instead of the traditional update
        self.register_buffer("embedding", embedding_init)

        # the followings are required for the EMA computation and the weights update
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_avg", embedding_init.clone())
    
    def quantize(self, encoding_indices):
        # since we created the embedding weights of shape (D, K), we need to transpose it to (K, D)
        # this is because nn.Embedding uses dimension (K, D)
        # .transpose(0, 1) will swap the first 2 dimension
        return F.embedding(encoding_indices, self.embedding.transpose(0, 1))

    def forward(self, x):
        # note: pytorch data shape == (batch size, channel, height, width)
        # here, we change the shape into (batc size, height, width, channel)
        # this is to make the shape follows the requirement of the next step
        x = x.permute(0, 2, 3, 1).contiguous()
        # note: contiguous() needs to be called after permute. Otherwise, the following view() will return an error
        
        # firstly, flatten all other dimension except for the last one (the channels dimension)
        # for example, in the sonnet documentation, input tensor of shape (16, 32, 32, 64)
        # will be reshaped to (16 * 32 * 32, 64)
        # which means we have 16 * 32 * 32 tensors of 64 dimensions
        # 64 here is the input parameter D in our implementation
        x_flatten = x.view(-1, self.D)

        # get the nearest embedding
        # by computing the distance between the encoder output and all embedding vector using the following formula
        # (z_ex - e_j)**2
        # where z_ex is the input/the encoding output and e_j is the embedding vector j
        # print(x_flatten.shape) # torch.Size([16384, 64])
        # print(self.embedding.shape) # torch.Size([64, 512])
        distance = (
            torch.sum(x_flatten ** 2, dim=1, keepdim=True) + # shape: torch.Size([16384, 1])
            torch.sum(self.embedding ** 2, dim=0, keepdim=True) - # shape: torch.Size([1, 512])
            # torch.Size([16384, 1]) + torch.Size([1, 512]), using python broadcasting: torch.Size([16384, 512]) + torch.Size([16384, 512])
            2 * torch.matmul(x_flatten, self.embedding) # shape: torch.Size([16384, 512])
        )
        # print(distance.shape) # torch.Size([16384, 512])
        # compute the distance between each of the 16384 input vector and 512 embedding vectors

        # for each of 16384 input vectors of size D, get the minimum distance
        # thus, out of K embedding vectors, choose 1 that gives us the minimum distance
        nearest_embedding = torch.argmin(distance, dim=1)
        # print(nearest_embedding.shape) # torch.Size([16384])

        return x

if __name__ == "__main__":
    tensor = torch.randn((16, 64, 32, 32))
    net = VectorQuantizerEMA(D=64, K=512)
    net(tensor)
    # print(net(tensor).shape)