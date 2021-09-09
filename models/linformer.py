import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# based on https://github.com/lucidrains/linformer

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class LinformerSelfAttention(nn.Module):
    def __init__(self, H):
        # Maybe don't share weights across all heads
        self.key = nn.Linear(H.bert_n_emb, H.bert_n_emb, bias=False)
        self.query = nn.Linear(H.bert_n_emb, H.bert_n_emb, bias=False)
        self.value = nn.Linear(H.bert_n_emb, H.bert_n_emb, bias=False)

        seq_len = np.prod(H.latent_shape)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, H.linformer_k)))
        self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, H.linformer_k)))


