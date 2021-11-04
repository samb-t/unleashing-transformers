# file for running the training of the VQGAN
import torch
import numpy as np
import copy
import deepspeed
import time
import random
from torchvision.transforms.functional import hflip
from models.vqgan import VQGAN
from hparams import get_vqgan_hparams
from utils.data_utils import get_data_loader, cycle
from utils.train_utils import EMA
from utils.log_utils import log, log_stats, save_model, save_stats, save_images, \
                            display_images, setup_visdom, config_log, start_training_log
from utils.vqgan_utils import unpack_vqgan_stats, load_vqgan_from_checkpoint, calc_FID
torch.backends.cudnn.benchmark = True

def main(H, vis):
    vqgan = VQGAN(H).cuda()
    # only load val_loader if running eval
    train_loader, val_loader = get_data_loader(
        H.dataset,
        H.img_size,
        H.batch_size,
        get_val_train_split=(H.steps_per_eval != 0)
    )
    train_iterator = cycle(train_loader)
    if val_loader != None:
        val_iterator = cycle(val_loader)


    if H.ema:
        ema = EMA(H.ema_beta)
        ema_vqgan = copy.deepcopy(vqgan)
    else:
        ema_vqgan = None
    
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


    losses = np.array([])
    mean_losses = np.array([])
    val_losses = np.array([])
    recon_losses = np.array([])
    latent_ids = []
    fids = np.array([])
    best_fid = float('inf')

    #NOTE this is getting messy now - need to tidy up all this steps stuff, easier to just build up a list of steps I think
    start_step = 0
    log_start_step = 0
    eval_start_step = H.steps_per_eval 
    if H.load_step > 0:
        start_step = H.load_step + 1 # don't repeat the checkpointed step
        vqgan, optim, d_optim, ema_vqgan, train_stats = load_vqgan_from_checkpoint(H, vqgan, optim, d_optim, ema_vqgan)

        # stats won't load for old models with no associated stats file
        if train_stats is not None: 
            losses, mean_losses, val_losses, latent_ids, fids, best_fid, H.steps_per_log, H.steps_per_eval = unpack_vqgan_stats(train_stats)
            log_start_step = 0
            eval_start_step = H.steps_per_eval
            log('Loaded stats')
        else:
            log('No stats found for model, displaying stats from load step instead')
            log_start_step = start_step
            if H.steps_per_eval:
                if H.steps_per_eval == 1:
                    eval_start_step = start_step
                else:
                    # set eval_start_step to nearest next eval point
                    # NOTE could just define an array to hold steps the whole time and add as it goes? more/less efficient? does it matter?
                        # would have to regenerate steps list again anyway
                    eval_start_step = start_step + H.steps_per_eval - start_step % H.steps_per_eval

    steps_per_epoch = len(train_loader)
    log(f'Epoch length: {steps_per_epoch}')

    print("ae", sum(p.numel() for p in vqgan.ae.parameters()))
    print("disc", sum(p.numel() for p in vqgan.disc.parameters()))
    print("total", sum(p.numel() for p in vqgan.ae.parameters()) + sum(p.numel() for p in vqgan.disc.parameters()))

    for step in range(start_step, H.train_steps):
        step_start_time = time.time()
        batch = next(train_iterator)

        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        
        if H.horizontal_flip: # randomly flip 50% of samples
            if random.random() <= 0.5:
            # vis.images(x, win='preflip', opts=dict(title='Images Pre-Flip'))
                x = hflip(x)
            # vis.images(x, win='postflip', opts=dict(title='Images Post-Flip'))

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
                d_scaler.step(d_optim)
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
            step_time = time.time() - step_start_time
            stats['step_time'] = step_time
            mean_losses = np.append(mean_losses, mean_loss)
            recon_losses = np.append(recon_losses, stats['l1'])
            losses = np.array([])

            vis.line(
                mean_losses, 
                list(range(log_start_step, step+1, H.steps_per_log)),
                win='loss',
                opts=dict(title='Loss')
            )
            # vis.line(
            #     recon_losses, 
            #     list(range(log_start_step, step+1, H.steps_per_log)),
            #     win='recon_loss',
            #     opts=dict(title='Train L1 Loss')
            # )
            log_stats(step, stats)         

        # bundled validation loss and FID calculations together
        # NOTE put in seperate function?
        if H.steps_per_eval:
            if step % H.steps_per_eval == 0 and step > 0:
                # log('Evaluating FIDs and validation loss:')
                # vqgan.eval()
                # # Calc FIDs
                # fid = calc_FID(H, ema_vqgan if H.ema else vqgan)
                # fids = np.append(fids, fid)
                # log(f'FID: {fid}')
                # if fid < best_fid:
                #     save_model(ema_vqgan if H.ema else vqgan, 'vqgan_bestfid', step, H.log_dir)

                # Calc validation losses
                x_val = next(val_iterator)
                if H.deepspeed:
                    x_val = x_val.half()
                _, val_stats = vqgan.val_iter(x, step)
                val_losses = np.append(val_losses, val_stats['l1'])

                steps = [step for step in range(eval_start_step, step+1, H.steps_per_eval)]
                # vis.line(fids, steps, win='FID',opts=dict(title='FID'))
                vis.line(val_losses, steps, win='val', opts=dict(title='Validation L1 Loss'))
                
                # vqgan.train()

        # log codebook usage
        if step % steps_per_epoch == 0 and step > 0: 
            latent_ids = torch.cat(latent_ids, dim=0)
            unique_code_ids = torch.unique(latent_ids).to(dtype=torch.int64)
            log(f'Codebook size: {H.codebook_size}   Unique Codes Used in Epoch: {len(unique_code_ids)}')            
            latent_ids = []

        if H.ema and step % H.steps_per_update_ema == 0 and step > 0:
            ema.update_model_average(ema_vqgan, vqgan)

        if step % H.steps_per_display_output == 0 and step > 0:
            display_images(vis, x, H, 'Original Images')
            # if H.ema:
            #     x_hat, _ = ema_vqgan.train_iter(x, step)
            x_hat = x_hat.detach().cpu().to(torch.float32)
            display_images(vis, x_hat, H, 'VQGAN Recons')
        
        if step % H.steps_per_save_output == 0:
            save_images(x_hat, 'recons', step, H.log_dir, H.save_individually)

        if step % H.steps_per_checkpoint == 0 and step > H.load_step:

            save_model(vqgan, 'vqgan', step, H.log_dir)#
            save_model(optim, 'ae_optim', step, H.log_dir)
            save_model(d_optim, 'disc_optim', step, H.log_dir)
            if H.ema:
                save_model(ema_vqgan, 'vqgan_ema', step, H.log_dir)

            train_stats = {
                'losses' : losses,
                'mean_losses' : mean_losses,
                'val_losses' : val_losses,
                'latent_ids' : latent_ids,
                'fids' : fids,
                'best_fid' : best_fid,
                'steps_per_log' : H.steps_per_log,
                'steps_per_eval' : H.steps_per_eval,
            }

            save_stats(H, train_stats, step)

if __name__=='__main__':
    H = get_vqgan_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for VQGAN on {H.dataset}')   
    start_training_log(H)
    main(H, vis)
