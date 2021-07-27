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

torch.backends.cudnn.benchmark = True


def get_sampler(H, ae, data_loader):
    # have to load in whole vqgan to retrieve ae values, garbage collector should disposed of unused params
    vqgan = VQGAN(ae, H)
    ae = load_model(vqgan, 'vqgan_ema', H.ae_load_step, H.ae_load_dir).ae
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

#TODO: break main() into more seperate functions to improve readability
#TODO: combine all model saving and loading (i.e. saving ae and disc as one object), maybe look at using checkpointing instead of saving and loading seperate components
def main(H, vis):
    ae = VQAutoEncoder(H) # TODO: we only actually need the decoder for latent recons, may as well remove half the params!
    
    # load vqgan (training stage 1)
    if H.model == 'vqgan':
        latent_ids = []
        H.batch_size = H.vqgan_batch_size
        data_loader = get_data_loader(H.dataset, H.img_size, H.batch_size, shuffle=True)
        test_data_loader = get_data_loader(H.dataset, H.img_size, H.batch_size, drop_last=False, shuffle=False)
        data_iterator = cycle(data_loader)
        model = VQGAN(ae, H).cuda()
        if H.deepspeed:
            model_engine, d_engine = model.ae_engine, model.d_engine
            optim, d_optim = model.optim, model.d_optim
        else:
            optim = torch.optim.Adam(model.ae.parameters(), lr=H.vqgan_lr)
            d_optim = torch.optim.Adam(model.disc.parameters(), lr=H.vqgan_lr)
    
    # load sampler (training stage 2)
    else:
        data_loader, data_iterator = get_latent_loaders(H, ae)
        model = get_sampler(H, ae, data_loader).cuda()
        if H.deepspeed:
            model_engine, optim, _, _ = deepspeed.initialize(args=H, model=model, model_parameters=model.parameters())
        else:
            optim = torch.optim.Adam(model.parameters(), lr=H.lr)

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_model = copy.deepcopy(model)

    if H.load_step > 0:
        model = load_model(model, H.model, H.load_step, H.load_dir).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_model = load_model(ema_model, f'{H.model}_ema', H.load_step, H.load_dir)
            except:
                ema_model = copy.deepcopy(model)
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
    steps_per_epoch = len(data_loader)
    fids = []
    best_fid = float('inf')
    scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    print('Epoch length:', steps_per_epoch)

    for step in range(start_step, H.train_steps):
        if H.warmup_iters:
            optim_warmup(H, step, optim)

        batch = next(data_iterator)
        if isinstance(batch, tuple):
            x, *target = batch
        else:
            x, target = batch, None
        x = x.cuda()
        if target is not None:
            target = target[0].cuda()
        
        if H.deepspeed:
            optim.zero_grad()
            if H.model == 'vqgan':
                x, target = x.half(), target.half() # TODO: Figure out this casting, only seems to be needed for VQGAN
            stats = model.train_iter(x, target, step)
            model_engine.backward(stats['loss'])
            model_engine.step()
        elif H.amp:
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                stats = model.train_iter(x, target, step)
            scaler.scale(stats['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            stats = model.train_iter(x, target, step)
            if torch.isnan(stats['loss']).any():
                print(f'Skipping step {step} with NaN loss')
                continue
            optim.zero_grad()
            stats['loss'].backward()
            optim.step()

        losses = np.append(losses, stats['loss'].item())

        if H.model == 'vqgan':
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

            # collect stats
            latent_ids.append(stats['latent_ids'].cpu().contiguous())

            # calculate FIDs 
            if (step % H.steps_per_fid_calc == 0 or step == start_step) and step > 0:
                if H.dataset == 'cifar10':
                    recons_epoch = collect_recons(H, ema_model if H.ema else model, test_data_loader)
                    recons_epoch = (recons_epoch * 255).clamp(0, 255).to(torch.uint8)
                    recons_epoch = TensorDataset(recons_epoch)
                    # TODO: just use test_data_loader instead, probably have to has to uint8 though
                    fid = torch_fidelity.calculate_metrics(input1=recons_epoch, input2='cifar10-train', 
                        cuda=True, fid=True, verbose=False)["frechet_inception_distance"]
                    fids.append(fid)
                    log(f'FID: {fid}')
                    vis.line(fids, win='FID',opts=dict(title='FID'))
                    if fid < best_fid:
                        save_model(ema_model if H.ema else model, f'{H.model}_bestfid', step, H.log_dir)

            if step % steps_per_epoch == 0 and step > 0: 
                latent_ids = torch.cat(latent_ids, dim=0)
                unique_code_ids = torch.unique(latent_ids).to(dtype=torch.int64)
                log(f'Codebook size: {H.codebook_size}   Unique Codes Used in Epoch: {len(unique_code_ids)}')
                
                # codebook recycling
                if H.quantizer == 'nearest' and H.code_recycling:
                    unused_code_ids = torch.tensor([code_id for code_id in range(H.codebook_size) if code_id not in unique_code_ids]).int()
                    model.ae.quantize.recycle_unused_codes(unique_code_ids, unused_code_ids)
                
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
            images, output_win_name = display_output(H, x, vis, data_iterator, ae, ema_model if H.ema else model)
            if step % H.steps_per_save_output == 0:
                save_images(images, output_win_name, step, H.log_dir, H.save_individuallyH)

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

