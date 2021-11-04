import torch
import torch.nn as nn
import torch.nn.functional as F
from .sampler import Sampler
from .transformer import Transformer
import numpy as np
import math

class AutoregressiveTransformer(Sampler):
    def __init__(self, H, embedding_weight):
        super().__init__(H, embedding_weight)
        self.net = Transformer(H)
        self.n_samples = H.batch_size
        self.seq_len = np.prod(H.latent_shape)
    
    def train_iter(self, x):
        x_in = x[:,:-1] # x is already flattened
        logits = self.net(x_in)
        loss = F.cross_entropy(logits.permute(0,2,1), x, reduction='none')
        loss = loss.sum(1).mean() / (math.log(2) * x.shape[1:].numel())
        stats = {'loss': loss}
        return stats

    def sample(self, temp=1.0):
        b, device = self.n_samples, 'cuda'
        x = torch.zeros(b, 0).long().to(device)
        for _ in range(self.seq_len):
            logits = self.net(x)[:, -1]
            probs = F.softmax(logits / temp, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, ix), dim=1)
        return x
        