# file for running the training of the VQGAN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from models import VQAutoEncoder, VQGAN, EBM, BERT, MultinomialDiffusion, SegmentationUnet, AbsorbingDiffusion, Transformer
from hparams import get_training_hparams
from utils import *
import torch_fidelity
import deepspeed


# TODO: move to utils
def optim_warmup(H, step, optim):
    if step <= H.warmup_iters:
        lr = H.lr * float(step) / H.warmup_iters
        for param_group in optim.param_groups:
            param_group['lr'] = lr

@torch.no_grad()
def display_output(H, x, vis, data_iterator, ae, model):            
    with torch.no_grad():
        if H.model == 'vqgan':
            images, *_ = model.ae(x)

            display_images(vis, x, H, win_name='Original Images')
            output_win_name = 'recons'

        else:

            latents = model.sample() #TODO need to write sample function for EBMS (give option of AIS?)
            q = model.embed(latents)
            images = ae.generator(q.cpu())
            output_win_name = 'samples'
            
        display_images(vis, images, H, win_name=output_win_name)

    return images, output_win_name