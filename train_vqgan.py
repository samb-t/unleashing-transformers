# file for running the training of the VQGAN
import torch
import numpy as np
import copy
from models import VQGAN
from hparams import get_vqgan_hparams
from utils import *
import deepspeed
torch.backends.cudnn.benchmark = True

def main(H, vis):
    vqgan = VQGAN(H).cuda()
    data_loader = get_data_loader(
        H.dataset,
        H.img_size,
        H.batch_size,torch.backends.cudnn.benchmark = True
        shuffle=True
    )

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_vqgan = copy.deepcopy(vqgan)
    else:
        ema_vqgan = None
    data_iterator = cycle(data_loader)
    
    # load vqgan (training stage 1)
    latent_ids = []
    if H.deepspeed:
        model_engine, optim, _, _ = deepspeed.initialize(
                                        args=H,
                                        model=vqgan.ae, 
                                        model_parameters=vqgan.ae.parameters()
                                    )
        d_engine, d_optim, _, _ =  deepspeed.initialize(
                                        args=H,
                                        model=vqgan.disc, 
                                        model_parameters=vqgan.disc.parameters()
                                    )
    else:
        optim = torch.optim.Adam(vqgan.ae.parameters(), lr=H.lr)
        d_optim = torch.optim.Adam(vqgan.disc.parameters(), lr=H.lr)

    if H.amp:
        scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

    if H.load_step > 0:
        vqgan, optim, d_optim, ema_vqgan = load_from_checkpoint(H, vqgan, optim, d_optim, ema_vqgan)

    losses = np.array([])
    mean_losses = np.array([])
    fids = []
    best_fid = float('inf')

    steps_per_epoch = len(data_loader)
    print('Epoch length:', steps_per_epoch)

    start_step = H.load_step # default 0
    for step in range(start_step, H.train_steps):
        batch = next(data_iterator)

        #TODO: fix bugs here, this doesn't seem to work for CIFAR (does it work for churches?)
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        
        x = x.cuda()
        if H.deepspeed:
            optim.zero_grad()
            x = x.half() # only seems to be needed for vqgan?
            x_hat, stats = vqgan.train_iter(x, step)
            model_engine.backward(stats['loss'])
            model_engine.step()
        elif H.amp:
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                x_hat, stats = vqgan.train_iter(x, step)
            scaler.scale(stats['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            x_hat, stats = vqgan.train_iter(x, step)
            optim.zero_grad()
            stats['loss'].backward()
            optim.step()

        losses = np.append(losses, stats['loss'].item())

        if step > H.disc_start_step:
            if H.deepspeed:
                d_optim.zero_grad()
                d_engine.backward(stats['d_loss'])
                d_engine.step()
            elif H.amp:
                d_optim.zero_grad()
                d_scaler.scale(stats['d_loss']).backward()
                d_scaler.step()
                d_scaler.update()
            else:
                d_optim.zero_grad()
                stats['d_loss'].backward()
                d_optim.step()

        # collect latent ids
        latent_ids.append(stats['latent_ids'].cpu().contiguous())

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

        # calculate FIDs 
        if (step % H.steps_per_calc_fid == 0 or step == start_step) and step > 0:
            fid = calc_FID(H, ema_vqgan if H.ema else vqgan)
            fids.append(fid)
            log(f'FID: {fid}')
            vis.line(fids, win='FID',opts=dict(title='FID'))
            if fid < best_fid:
                save_model(ema_vqgan if H.ema else vqgan, 'vqgan_bestfid', step, H.log_dir)

        # log codebook usage
        if step % steps_per_epoch == 0 and step > 0: 
            latent_ids = torch.cat(latent_ids, dim=0)
            unique_code_ids = torch.unique(latent_ids).to(dtype=torch.int64)
            log(f'Codebook size: {H.codebook_size}   Unique Codes Used in Epoch: {len(unique_code_ids)}')            
            latent_ids = []

        if H.ema and step % H.steps_per_update_ema == 0 and step > 0:
            # log(f'Updating ema for step {step}')
            ema.update_model_average(ema_vqgan, vqgan)

        if step % H.steps_per_display_output == 0 and step > 0:
            display_images(vis, x, H, 'Original Images')
            display_images(vis, x_hat, H, 'VQGAN Recons')
        
        if step % H.steps_per_save_output == 0:
            save_images(x_hat, 'recons', step, H.log_dir, H.save_individually)

        if step % H.steps_per_checkpoint == 0 and step > H.load_step:

            save_model(vqgan, 'vqgan', step, H.log_dir)#
            save_model(optim, 'ae_optim', step, H.log_dir)
            save_model(d_optim, 'disc_optim', step, H.log_dir)
            if H.ema:
                save_model(ema_vqgan, 'vqgan_ema', step, H.log_dir)


if __name__=='__main__':
    H = get_vqgan_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for VQGAN on {H.dataset}')   
    start_training_log(H)
    main(H, vis)
