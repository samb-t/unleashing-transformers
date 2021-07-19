# annealed importance sampling for GWG EBM model, modified from:
# https://github.com/wgrathwohl/GWG_release/blob/main/ais.py

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models import EBM, VQAutoEncoder
from utils import *
from hparams import get_ais_hparams

class AISModel(nn.Module):
    def __init__(self, model, init_dist):
        super().__init__()
        self.ebm = model
        self.init_dist = init_dist

    def forward(self, x, beta):
        logpx = self.model(x).squeeze()
        logpi = self.init_dist.log_prob(x).sum(-1)
        return logpx * beta + logpi * (1. - beta)


def set_up_AIS(H):
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

    ae = load_model(ae, 'ae', H.ae_load_step, H.ae_load_dir)
    embedding_weight = ae.quantize.embedding.weight
    data_loader, _ = get_latent_loaders(H, ae)
    init_dist = get_init_dist(H, data_loader)
    ebm = EBM(H, embedding_weight).cuda()
    ebm = load_model(ebm, 'ebm', H.load_step, H.ebm_load_dir)

    model = AISModel(ebm, init_dist)
    model = model.cuda()

    return ae, model, init_dist


def main(H, vis):
    ae, model, init_dist = set_up_AIS(H)
    betas = np.linspace(0., 1., H.ais_iters)
    samples = init_dist.sample((H.n_samples,))
    log_w = torch.zeros((H.n_samples,)).cuda()

    for itr, beta_k in tqdm(enumerate(betas)):
        if itr == 0:
            continue

        beta_km1 = betas[itr - 1]

        # update importance weights
        with torch.no_grad():
            log_w = log_w + model(samples, beta_k) - model(samples, beta_km1)

        model_k = lambda x: model(x, beta=beta_k)

        for _ in range(H.steps_per_AIS_iter):
            samples = model.ebm.gibbs_sampler.step(samples.detach(), model_k).detach()

        if itr % H.steps_per_display_output == 0:
            latent_ids = samples.max(2)[1].detach()
            q = model.ebm.embed(latent_ids)
            images = ae.generator(q)
            display_images(vis, images, H, 'AIS_Samples')

            if itr % H.steps_per_save_output == 0:
                log(f'Saving samples at AIS step {itr}')
                save_images(images, 'ais_samples', itr, H.log_dir)


if __name__=='__main__':
    H = get_ais_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir, filename='sampling_log.txt')
    start_training_log(H)
    log('---------------------------------')
    main(H, vis)