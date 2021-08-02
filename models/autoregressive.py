import torch
import torch.nn as nn
from .sampler import Sampler
from .transformer import Transformer

class AutoregressiveTransformer(Sampler):
    def __init__(self, H, embedding_weight):
        super().__init__(H, embedding_weight)
        self.net = Transformer(H)

    
    def train_iter(self, x):
        # x is already flattened
        x_target = 