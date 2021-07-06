from shutil import ExecError
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import VQAutoEncoder, VQGAN, EBM, BERT
from hparams import get_hparams
from utils import *

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
    )
    
    start_step = 0
    if H.model != 'vqgan':
        if not H.ae_load_step:
            raise KeyError('Please sepcificy an autoencoder to load using the --ae_load_step flag')
        
        ae = load_model(ae, 'ae', H.ae_load_step, f'vqgan_{H.dataset}_{H.latent_shape[-1]}')
        data_loader, data_iterator = get_latent_loaders(H, ae)
        init_dist = get_init_dist(H, data_loader)

        if H.model == 'ebm':
            if H.load_step > 0:
                buffer = load_buffer(H.load_step, H.log_dir)
            else:
                buffer = []
                for _ in range(int(H.buffer_size / 100)):
                    buffer.append(init_dist.sample((100,)).max(2)[1].cpu())
                buffer = torch.cat(buffer, dim=0)
            model = EBM(H, ae.quantize.embedding.weight, buffer).cuda()


        elif H.model == 'bert':
            # TODO: change this to pass in H instead of all the args
            model = BERT(H, ae.quantize.embedding).cuda()

        elif H.model == 'diffusion':
            ...
        
        optim = torch.optim.Adam(model.parameters(), lr=H.lr)
        if H.load_step > 0:
            model = load_model(model, H.model, H.load_step, H.log_dir)
            if H.load_optim:
                optim = load_model(optim, f'{H.model}_optim', H.load_step, H.log_dir)
                for param_group in optim.param_groups:
                    param_group['lr'] = H.lr
            
    else:
        data_iterator = cycle(get_data_loader(H.dataset, H.img_size, H.vqgan_batch_size))
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

    start_step = H.load_step
    for step in range(start_step, H.train_steps):
        if H.warmup_iters:
            if step < H.warmup_iters:
                lr = H.lr * float(step) / H.warmup_iters
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

        # TODO: replace this to make it same for all models

        else:
            x, *target = next(data_iterator)
            x = x.cuda()
            if target:
                target = target[0].cuda()
            stats = model.train_iter(x, target, step)

        optim.zero_grad()
        stats['loss'].backward()
        optim.step()

        if H.model == 'vqgan' and step > H.disc_start_step:
            d_optim.zero_grad()
            stats['d_loss'].backward()
            d_optim.step()

        log_stats(step, stats)

        if step % H.steps_per_display_output == 0 and step > 0:
            display_images(vis, stats['images'], H)

            if step % H.steps_per_save_output == 0:
                save_images(stats['images'], 'samples', step, H.log_dir)

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
    main(H, vis)
