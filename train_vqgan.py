# file for running the training of the VQGAN
import torch
import numpy as np
import copy

from visdom import Tensor
from models import VQGAN
from hparams import get_vqgan_hparams
from utils import *
import torch_fidelity
import deepspeed

def load_from_checkpoint(H, vqgan, optim, d_optim, ema_vqgan):
    vqgan = load_model(vqgan, 'vqgan', H.load_step, H.load_dir).cuda()
    if H.ema:
        try:
            ema_vqgan = load_model(
                            ema_vqgan,
                            f'vqgan_ema',
                            H.load_step, 
                            H.load_dir
                        )
        except:
            log('No ema model found at checkpoint, starting from loaded model')
            ema_vqgan = copy.deepcopy(vqgan)
    if H.load_optim:
            optim = load_model(optim, f'ae_optim', H.load_step, H.load_dir)
            d_optim = load_model(d_optim, 'disc_optim', H.load_step, H.load_dir)

    return vqgan, optim, d_optim, ema_vqgan


def main(H, vis):
    vqgan = VQGAN(H).cuda()
    data_loader = get_data_loader(
        H.dataset,
        H.img_size,
        H.batch_size,
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
        optim = torch.optim.Adam(vqgan.ae.parameters(), lr=H.vqgan_lr)
        d_optim = torch.optim.Adam(vqgan.disc.parameters(), lr=H.vqgan_lr)

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

    start_step = H.load_step # defaults to 0 if not specified
    for step in range(start_step, H.train_steps):
        if H.warmup_iters:
            optim_warmup(H, step, optim)

        batch = next(data_iterator)

        #TODO: fix bugs here, this doesn't seem to work for CIFAR (does it work for churches?)
        if isinstance(batch, tuple):
            x, *target = batch
        else:
            x, target = batch, None
        # x = x.cuda()
        x = x[0].cuda()
        if target is not None:
            target = target[0].cuda()
        
        if H.deepspeed:
            optim.zero_grad()
            # x, target = x.half(), target.half() 
            x = x.half() # only seems to be needed for vqgan?
            stats = vqgan.train_iter(x, target, step)
            model_engine.backward(stats['loss'])
            model_engine.step()
        elif H.amp:
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                stats = vqgan.train_iter(x, target, step)
            scaler.scale(stats['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            stats = vqgan.train_iter(x, target, step)
            # NOTE: shouldn't be getting NaN loss, likely to be removed
            # if torch.isnan(stats['loss']).any():
            #     print(f'Skipping step {step} with NaN loss')
            #     continue
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

        # calculate FIDs 
        if (step % H.steps_per_fid_calc == 0 or step == start_step) and step > 0:
            if H.dataset == 'cifar10':
                images, recons = collect_ims_and_recons(H, ema_vqgan if H.ema else vqgan)
                images = TensorDataset(images)
                recons = TensorDataset(recons)
                fid = torch_fidelity.calculate_metrics(input1=recons, input2=images, 
                    cuda=True, fid=True, verbose=True)["frechet_inception_distance"]
                fids.append(fid)
                log(f'FID: {fid}')
                vis.line(fids, win='FID',opts=dict(title='FID'))
                if fid < best_fid:
                    save_model(ema_vqgan if H.ema else vqgan, f'{H.model}_bestfid', step, H.log_dir)

        if step % steps_per_epoch == 0 and step > 0: 
            latent_ids = torch.cat(latent_ids, dim=0)
            unique_code_ids = torch.unique(latent_ids).to(dtype=torch.int64)
            log(f'Codebook size: {H.codebook_size}   Unique Codes Used in Epoch: {len(unique_code_ids)}')            
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
            ema.update_model_average(ema_vqgan, vqgan)

        if step % H.steps_per_display_output == 0 and step > 0:
            images, output_win_name = display_output(H, x, vis, data_iterator, ae, ema_vqgan if H.ema else model)
            if step % H.steps_per_save_output == 0:
                save_images(images, output_win_name, step, H.log_dir, H.save_individuallyH)

        # TODO: merge VQGAN and Sampler saving / loading (maybe)
        if step % H.steps_per_checkpoint == 0 and step > H.load_step:

            save_model(vqgan, H.model, step, H.log_dir)
            save_model(optim, 'ae_optim', step, H.log_dir)
            save_model(d_optim, 'disc_optim', step, H.log_dir)

            if H.ema:
                save_model(ema_vqgan, f'{H.model}_ema', step, H.log_dir)


if __name__=='__main__':
    H = get_vqgan_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for VQGAN on {H.dataset}')   
    start_training_log(H)
    main(H, vis)
