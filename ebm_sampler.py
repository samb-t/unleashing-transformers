#%% imports
from energy import AE_LOAD_STEP, latent_ids_to_onehot
import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
from torch.utils import data
import visdom
import os
import time
from utils import *
from hparams import Hparams
from torch.nn.utils import parameters_to_vector as ptv
from vqgan_new import VQAutoEncoder, ResBlock, VectorQuantizer, Encoder, Generator
from energy import ResNetEBM_cat, EBM, DiffSamplerMultiDim, MyOneHotCategorical
from tqdm import tqdm


dataset = 'celeba'
log_dir = f'ebm_{dataset}'
H = Hparams(dataset)

AE_LOAD_STEP = 470000
EBM_LOAD_STEP = 18000

data_dim = np.prod(H.latent_shape)
sampler = DiffSamplerMultiDim(data_dim, 1)

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
ae = load_model(ae, 'ae', AE_LOAD_STEP, f'vqgan_{dataset}')

latent_ids = torch.load(f'latents/{dataset}_latents')
latent_loader = torch.utils.data.DataLoader(latent_ids, batch_size=H.batch_size, shuffle=False)
latent_iterator = cycle(latent_loader)

init_dist = torch.load(f'latents/{dataset}_init_dist')
init_mean = init_dist.mean

net = ResNetEBM_cat(H.emb_dim)
energy = EBM(
    net,
    ae.quantize.embedding,
    H.codebook_size,
    H.emb_dim,
    H.latent_shape,
    mean=init_mean,
).cuda()

energy = load_model(energy, 'ebm', EBM_LOAD_STEP, log_dir)
buffer = load_buffer(EBM_LOAD_STEP, log_dir)

all_inds = list(range(H.buffer_size))

x = next(latent_iterator)
x = x.cuda()
x = latent_ids_to_onehot(x, H)


buffer_inds = sorted(np.random.choice(all_inds, H.batch_size, replace=False))
x_buffer = buffer[buffer_inds].cuda()
x_fake = x_buffer

for k in range(H.sampling_steps):
    x_fake_new = sampler.step(x_fake.detach(), energy).detach()
    x_fake = x_fake_new

q = energy.embed(x_fake)
samples = ae.generator(q)

vis = visdom.Visdom()
save_images(samples[:64], vis, 'samples', EBM_LOAD_STEP, log_dir)