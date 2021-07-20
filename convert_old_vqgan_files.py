import torch
from models import VQAutoEncoder, VQGAN
from hparams import get_training_hparams
from utils import *

def main(H):
    ae = VQAutoEncoder(
        H.n_channels,
        H.nf,
        H.res_blocks, 
        H.codebook_size, 
        H.emb_dim,
        H.ch_mult, 
        H.img_size, 
        H.attn_resolutions
    )
    vqgan = VQGAN(ae, H)
    vqgan.ae = load_model(vqgan.ae, 'ae', H.load_step, H.load_dir)
    vqgan.disc = load_model(vqgan.disc, 'discriminator', H.load_step, H.load_dir)

    save_model(vqgan, 'vqgan', H.load_step, H.load_dir)

    print('model converted')

if __name__=='__main__':
    H = get_training_hparams()
    main(H)
