import torch
import torch.nn as nn

class Sampler(nn.Module):
    def __init__(self, H, embedding_weight):
        super().__init__()
        self.latent_shape = H.latent_shape
        self.emb_dim = H.emb_dim
        self.codebook_size = H.codebook_size
        self.embedding_weight = embedding_weight

    def train_iter(self, x, step):
        raise NotImplementedError()

    def sample(self, n_samples):
        raise NotImplementedError()

    def class_conditional_train_iter(self, x, y):
        raise NotImplementedError()

    def class_conditional_sample(n_samples, y):
        raise NotImplementedError()

    def embed(self, z):
        z_flattened = z.view(-1, self.codebook_size) # B*H*W, codebook_size
        return torch.matmul(z_flattened, self.embedding_weight).view(
            z.size(0), 
            self.latent_shape[1],
            self.latent_shape[2],
            self.emb_dim
        ).permute(0, 3, 1, 2).contiguous()