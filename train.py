from shutil import ExecError
from models.vqgan import VQAutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import EBM, VQGAN
from hparams import get_hparams
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
    
    start_step = 0
    if H.model != 'vqgan':
        if not H.ae_load_step:
            raise KeyError('Please sepcificy an autoencoder to load using the --ae_load_step flag')
        ae = load_model(ae, 'ae', H.ae_load_step, f'vqgan_{H.dataset}_{H.latent_shape[-1]}')

        '''
        TODO: 
        - update hparam usage so that EBM comes with its own get_hparams() function that overwrites vqgan arguments
            - perhaps using code from here: https://stackoverflow.com/questions/46310720/replacing-argument-in-argparse
        - write load_ebm function in model_utils
        - rearrange such that there is a final if statement to generate and set buffer
            - only after model has been loaded
        '''
        if H.model == 'ebm':
            # TODO: refactor and simplify this, gets messy with optims / buffers etc.
            data_loader, data_iterator = get_latent_loaders(H, ae)
            init_dist = get_init_dist(H, data_loader)
            model = EBM(H, ae.quantize.embedding.weight).cuda()
            optim = torch.optim.Adam(model.parameters(), lr=H.lr)

            if H.load_step > 0:
                buffer = load_buffer(H.load_step, H.log_dir)
                model = load_model(model, 'ebm', H.load_step, H.log_dir)
                if H.load_optim:
                    optim = load_model(optim, 'ebm_optim', H.load_step, H.log_dir)
                    for param_group in optim.param_groups:
                        param_group['lr'] = H.ebm_lr
                start_step = H.load_step
            
            else:
                buffer = []
                for _ in range(int(H.buffer_size / 100)):
                    buffer.append(init_dist.sample((100,)).max(2)[1].cpu())
                buffer = torch.cat(buffer, dim=0)
            model.set_buffer(buffer)

    else:
        # TEMPORARY:
        data_iterator = cycle(get_data_loader(H.dataset, H.img_size, H.vqgan_batch_size))
        ##
        model = VQGAN(ae, H).cuda()
        optim = torch.optim.Adam(model.ae.parameters(), lr=H.vqgan_lr)
        d_optim = torch.optim.Adam(model.disc.parameters(), lr=H.vqgan_lr)


        if H.load_step > 0:
            model.ae = load_model(model.ae, 'ae', H.load_step, H.log_dir)
            model.disc = load_model(model.disc, 'discriminator', H.load_step, H.log_dir)
            if H.load_optim:
                optim = load_model(optim, f'ae_optim', H.load_step, H.log_dir)
                d_optim = load_model(d_optim, 'disc_optim', H.load_step, H.log_dir)
            start_step = H.load_step            

    print('loaded models')
    for step in range(start_step, H.train_steps):
        if H.warmup_iters:
            if step < H.warmup_iters:
                lr = H.lr * float(step) / H.warmup_iters
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
    
        x, *_ = next(data_iterator)
        x = x.cuda()
        stats = model.train_iter(x, step)

        optim.zero_grad()
        stats['loss'].backward()
        optim.step()

        if H.model == 'vqgan' and step > H.disc_start_step:
            d_optim.zero_grad()
            stats['d_loss'].backward()
            d_optim.step()

        log_stats(step, stats)

        # TODO: merge VQGAN and Sampler saving / loading 
        if step % H.steps_per_checkpoint == 0 and step > H.load_step:


            if H.model == 'vqgan':
                save_model(model.ae, 'ae', step, H.log_dir)
                save_model(model.disc, 'discriminator', step, H.log_dir)
                save_model(optim, 'ae_optim', step, H.log_dir)
                save_model(d_optim, 'disc_optim', step, H.log_dir)
            else:
                save_model(model, H.model, step, H.log_dir)
                save_model(optim, f'{H.model}_optim', step, H.log_dir)
                save_buffer(buffer, step, H.log_dir)


if __name__=='__main__':
    H = get_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.model}')   
    if H.model =='vqgan':
        start_training_log(H.get_vqgan_param_dict())
    else:
        start_training_log(H)
    main(H)