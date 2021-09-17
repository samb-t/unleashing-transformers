# file for running the training of the VQGAN
import torch
import torch.nn as nn
import numpy as np
import copy
import deepspeed
import time
from models.vqgan import VQGAN
from hparams import get_vqgan_hparams
from utils.data_utils import get_data_loader, cycle
from utils.train_utils import EMA
from utils.log_utils import log, log_stats, save_model, save_stats, save_images, \
                            display_images, setup_visdom, config_log, start_training_log, load_model
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

    vqgan = load_model(vqgan, 'vqgan', H.load_step, H.load_dir).cuda()
    ema_vqgan = load_model(ema_vqgan, 'vqgan_ema', H.load_step, H.load_dir)

    vqgan.ae.generator.logsigma = nn.Conv2d(vqgan.ae.generator.final_block_ch, H.n_channels, kernel_size=3, stride=1, padding=1).cuda()
    ema_vqgan.ae.generator.logsigma = nn.Conv2d(vqgan.ae.generator.final_block_ch, H.n_channels, kernel_size=3, stride=1, padding=1).cuda()
    ema_vqgan.ae.generator.logsigma.weight.data = vqgan.ae.generator.logsigma.weight.data
    ema_vqgan.ae.generator.logsigma.bias.data = vqgan.ae.generator.logsigma.bias.data

    optim = torch.optim.Adam(vqgan.ae.generator.logsigma.parameters(), lr=1e-4)

    # optim = torch.optim.SGD(vqgan.ae.generator.logsigma.parameters(), lr=0.001)

    if H.amp:
        scaler = torch.cuda.amp.GradScaler()

    losses = np.array([])
    mean_losses = np.array([])
    val_losses = np.array([])
    best_val_loss = float('inf')

    for step in range(H.train_steps):
        step_start_time = time.time()
        batch = next(train_iterator)

        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        
        x = x.cuda()

        if H.amp:
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                x_hat, stats = vqgan.probabilistic(x, step)
            scaler.scale(stats['nll']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            x_hat, stats = vqgan.probabilistic(x, step)
            optim.zero_grad()
            if torch.isnan(stats['nll']):
                print("skipping step")
                continue
            stats['nll'].backward()
            torch.nn.utils.clip_grad_norm_(vqgan.ae.generator.logsigma.parameters(), 0.1)
            optim.step()

        losses = np.append(losses, stats['nll'].item())

        if step % H.steps_per_log == 0:
            mean_loss = np.mean(losses)
            stats['loss'] = mean_loss
            step_time = time.time() - step_start_time
            stats['step_time'] = step_time
            mean_losses = np.append(mean_losses, mean_loss)
            losses = np.array([])
            vis.line(
                mean_losses, 
                # list(range(0, step+1, H.steps_per_log)),
                win='loss',
                opts=dict(title='Loss')
            )
            log_stats(step, stats)
        
        if H.steps_per_eval:
            if step % H.steps_per_eval == 0 and step > 0:
                # Calc validation losses
                val_loss, num_elems = 0.0, 0
                for x_val in val_loader:
                    with torch.no_grad():
                        x_val_hat, val_stats = vqgan.probabilistic(x_val[0].cuda(), step)
                    val_loss += val_stats['nll']*x_val[0].size(0)
                    num_elems += x_val[0].size(0)
                
                val_loss = val_loss.item()/num_elems
                
                val_losses = np.append(val_losses, val_loss)
                steps = [step for step in range(H.steps_per_eval, step+1, H.steps_per_eval)]
                vis.line(val_losses, win='val', opts=dict(title='Validation L1 Loss'))
                display_images(vis, x_val[0], H, 'Validation Images')
                display_images(vis, x_val_hat, H, 'Validation Recons')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(vqgan, 'vqgan', 0, H.log_dir)
                    if H.ema:
                        save_model(ema_vqgan, 'vqgan_ema', 0, H.log_dir)
        
        if H.ema and step % H.steps_per_update_ema == 0 and step > 0:
            ema.update_model_average(ema_vqgan, vqgan)

        if step % H.steps_per_display_output == 0 and step > 0:
            display_images(vis, x, H, 'Original Images')
            x_hat = x_hat.detach().cpu().to(torch.float32)
            display_images(vis, x_hat, H, 'VQGAN Recons')
        
        # if step % H.steps_per_checkpoint == 0 and step > H.load_step:

        #     save_model(vqgan, 'vqgan', step, H.log_dir)#
        #     save_model(optim, 'ae_optim', step, H.log_dir)
        #     if H.ema:
        #         save_model(ema_vqgan, 'vqgan_ema', step, H.log_dir)

        #     train_stats = {
        #         'losses' : losses,
        #         'mean_losses' : mean_losses,
        #         'val_losses' : val_losses,
        #         'steps_per_log' : H.steps_per_log,
        #         'steps_per_eval' : H.steps_per_eval,
        #     }

        #     save_stats(H, train_stats, step)


if __name__=='__main__':
    H = get_vqgan_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for VQGAN on {H.dataset}')   
    start_training_log(H)
    main(H, vis)
