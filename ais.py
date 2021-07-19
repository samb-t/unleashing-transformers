import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models import EBM, VQAutoEncoder
from utils import *

class AISModel(nn.Module):
    def __init__(self, model, init_dist):
        super().__init__()
        self.model = model
        self.init_dist = init_dist

    def forward(self, x, beta):
        logpx = self.model(x).squeeze()
        logpi = self.init_dist.log_prob(x).sum(-1)
        return logpx * beta + logpi * (1. - beta)


def main(H, vis):
    ae = VQAutoEncoder(
        H.n_channels,
        H.nf,
        H.res_blocks, 
        H.codebook_size, 
        H.emb_dim,
        H.ch_mult, 
        H.img_size, 
        H.attn_resolutions
    ).cuda()

    ae = load_model(ae, 'ae', H.ae_load_step, f'vqgan_{H.dataset}_{H.latent_shape[-1]}')
    embedding_weight = ae.quantize.embedding.weight
    data_loader, data_iterator = get_latent_loaders(H, ae)
    init_dist = get_init_dist(H, data_loader)

    model = EBM(H, embedding_weight).cuda()

    '''
    - [ ] load model and set up models
        - [x] AE
        - [ ] EBM
        - [ ] AISModel
    - [ ] run sampling on latents
    - [ ] convert latents from one hots to IDs
    - [ ] embed latents
    - [ ] pass through autoencoder to generate samples
    '''

if __name__=='__main__':
