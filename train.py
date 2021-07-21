import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from models import VQAutoEncoder, VQGAN, EBM, BERT, MultinomialDiffusion, SegmentationUnet, AbsorbingDiffusion, Transformer
from hparams import get_training_hparams
from utils import *


def get_sampler(H, ae, data_loader):
    # have to load in whole vqgan to retrieve ae values, garbage collector should disposed of unused params
    vqgan = VQGAN(ae, H)
    ae = load_model(vqgan, 'vqgan', H.ae_load_step, H.ae_load_dir).ae
    embedding_weight = ae.quantize.embedding.weight

    if H.model != 'diffusion':
        init_dist = get_init_dist(H, data_loader)

    if H.model == 'ebm':
        # TODO put buffer loading in seperate function
        if H.load_step > 0:
            buffer = load_buffer(H.load_step, H.log_dir)
        else:
            buffer = []
            for _ in range(int(H.buffer_size / 100)):
                buffer.append(init_dist.sample((100,)).max(2)[1].cpu())
            buffer = torch.cat(buffer, dim=0)
        model = EBM(H, embedding_weight, buffer)

    elif H.model == 'bert':
        model = BERT(H, embedding_weight)

    elif H.model == 'diffusion':
        if H.diffusion_net == 'unet':
            denoise_fn = SegmentationUnet(H)
        # create multinomial diffusion model
        model = MultinomialDiffusion(H, embedding_weight, denoise_fn)

    elif H.model == 'absorbing':
        denoise_fn = Transformer(H).cuda()
        model = AbsorbingDiffusion(H, denoise_fn, H.codebook_size, embedding_weight)

    return model


def optim_warmup(H, step, optim):
    if step <= H.warmup_iters:
        lr = H.lr * float(step) / H.warmup_iters
        for param_group in optim.param_groups:
            param_group['lr'] = lr


@torch.no_grad()
def display_output(H, vis, data_iterator, ae, model):            
    with torch.no_grad():
        if H.model == 'vqgan':
            x, *_ = next(data_iterator)
            x = x.cuda()
            images, *_ = model.ae(x)

            display_images(vis, x, H, win_name='Original Images')
            output_win_name = 'recons'

        else:

            latents = model.sample() #TODO need to write sample function for EBMS (give option of AIS?)
            q = model.embed(latents)
            images = ae.generator(q)
            output_win_name = 'samples'
            
        display_images(vis, images, H, win_name=output_win_name)

    return images, output_win_name

#TODO: break main() into more seperate functions to improve readability
#TODO: combine all model saving and loading (i.e. saving ae and disc as one object), maybe look at using checkpointing instead of saving and loading seperate components
def main(H, vis):
    ae = VQAutoEncoder(H)
    
    # load vqgan (training stage 1)
    if H.model == 'vqgan':
        latent_ids = []
        data_iterator = cycle(get_data_loader(H.dataset, H.img_size, H.vqgan_batch_size))
        model = VQGAN(ae, H).cuda()
        optim = torch.optim.Adam(model.ae.parameters(), lr=H.vqgan_lr)
        d_optim = torch.optim.Adam(model.disc.parameters(), lr=H.vqgan_lr)
    
    # load sampler (training stage 2)
    else:
        data_loader, data_iterator = get_latent_loaders(H, ae)
        model = get_sampler(H, ae, data_loader).cuda()
        optim = torch.optim.Adam(model.parameters(), lr=H.lr)

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_model = copy.deepcopy(model)

    if H.load_step > 0:
        model = load_model(model, H.model, H.load_step, H.load_dir).cuda()
        if H.ema:
            ema_model = load_model(ema_model, f'{H.model}_ema', H.load_step, H.load_dir)
        if H.load_optim:
            if H.model == 'vqgan':
                optim = load_model(optim, f'ae_optim', H.load_step, H.load_dir)
                d_optim = load_model(d_optim, 'disc_optim', H.load_step, H.load_dir)
            else:
                optim = load_model(optim, f'{H.model}_optim', H.load_step, H.load_dir)
                # only used when changing learning rates when reloading from checkpoint
                for param_group in optim.param_groups:
                    param_group['lr'] = H.lr         

    start_step = H.load_step # defaults to 0 if not specified
    losses = np.array([])
    mean_losses = np.array([])

    for step in range(start_step, H.train_steps):
        if H.warmup_iters:
            optim_warmup(H, step, optim)

        x, *target = next(data_iterator)
        x = x.cuda()
        if target:
            target = target[0].cuda()
        
        stats = model.train_iter(x, target, step)

        optim.zero_grad()
        stats['loss'].backward()
        optim.step()
        losses = np.append(losses, stats['loss'].item())

        if H.model == 'vqgan':
            if step > H.disc_start_step:
                d_optim.zero_grad()
                stats['d_loss'].backward()
                d_optim.step()

            latent_ids.append(stats['latent_ids'].cpu().contiguous())
            if step % 100 == 0: # TODO; change to once per epoch
                latent_ids = torch.cat(latent_ids, dim=0)
                unique_codes_count = len(torch.unique(latent_ids))
                log(f'Codebook size: {H.codebook_size}   Unique Codes: {unique_codes_count}')
                latent_ids = []

        if step % H.steps_per_log == 0:
            mean_loss = np.mean(losses)
            stats['loss'] = mean_loss
            mean_losses = np.append(mean_losses, mean_loss)
            losses = np.array([])
            vis.line(
                mean_losses, 
                list(range(start_step, step+1, H.steps_per_log)),
                win='loss',
                opts=dict(title='Loss')
            )
            log_stats(step, stats)

        if H.ema and step % H.steps_per_update_ema == 0 and step > 0:
            # log(f'Updating ema for step {step}')
            ema.update_model_average(ema_model, model)

        if step % H.steps_per_display_output == 0 and step > 0:
            images, output_win_name = display_output(H, vis, data_iterator, ae, ema_model if H.ema else model)
            if step % H.steps_per_save_output == 0:
                        save_images(images, output_win_name, step, H.log_dir)

        # TODO: merge VQGAN and Sampler saving / loading (maybe)
        if step % H.steps_per_checkpoint == 0 and step > H.load_step:

            save_model(model, H.model, step, H.log_dir)
            if H.ema:
                save_model(ema_model, f'{H.model}_ema', step, H.log_dir)
            if H.model == 'vqgan':
                save_model(optim, 'ae_optim', step, H.log_dir)
                save_model(d_optim, 'disc_optim', step, H.log_dir)
            else:
                save_model(optim, f'{H.model}_optim', step, H.log_dir)
            # if H.model == 'ebm':
            #     save_buffer(buffer, step, H.log_dir)


if __name__=='__main__':
    H = get_training_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.model}')   
    start_training_log(H)
    main(H, vis)
