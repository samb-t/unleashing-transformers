from models.vqgan import VQAutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from models import \
    MyOneHotCategorical, VQAutoEncoder, Generator,VQGAN,\
    EBM, BERT, MultinomialDiffusion, SegmentationUnet, \
    AbsorbingDiffusion, Transformer
from hparams import get_sampler_hparams
from utils import *
import deepspeed

torch.backends.cudnn.benchmark = True

# TODO: review ae loading and move this to utils
def get_sampler(H, embedding_weight, latent_loader):

    if H.sampler == 'absorbing':
        denoise_fn = Transformer(H).cuda()
        sampler =  AbsorbingDiffusion(H, denoise_fn, H.codebook_size, embedding_weight)

    elif H.sampler == 'bert':
        sampler = BERT(H, embedding_weight)

    elif H.sampler == 'diffusion':
        if H.diffusion_net == 'unet':
            denoise_fn = SegmentationUnet(H)
        sampler =  MultinomialDiffusion(H, embedding_weight, denoise_fn)

    elif H.sampler == 'ebm':
        # UNTESTED - need to fix EBM training first!
        init_mean = get_init_mean(H, latent_loader)
        init_dist = MyOneHotCategorical(init_mean)
        # TODO put buffer loading in seperate function
        if H.load_step > 0:
            buffer = load_buffer(H.load_step, H.log_dir)
        else:
            buffer = []
            for _ in range(int(H.buffer_size / 100)):
                buffer.append(init_dist.sample((100,)).max(2)[1].cpu())
            buffer = torch.cat(buffer, dim=0)
        sampler = EBM(H, embedding_weight, buffer)

    return sampler


def main(H, vis):
    
    latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_latents'
    if not os.path.exists(latents_filepath):        
        ae_state_dict = retrieve_autoencoder_components_state_dicts(H, ['encoder', 'quantize', 'generator'])
        ae = VQAutoEncoder(H)
        ae.load_state_dict(ae_state_dict)
        full_dataloader = get_data_loader(H.dataset, H.img_size, H.batch_size, drop_last=False, shuffle=False)
        ae = ae.cuda() # put ae on GPU for generating
        generate_latent_ids(H, ae, full_dataloader)
        #TODO: test if this actually frees up GPU space or not
        ae = ae.cpu()
        # del ae
    
    latent_loader = get_latent_loader(H, latents_filepath)
        
    # here is where to load generator and quantizer
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )

    embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight')
    if H.deepspeed:
        embedding_weight = embedding_weight.half()
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)
    # want generator on GPU? maybe add flag to save space if needbe
    generator.load_state_dict(quanitzer_and_generator_state_dict)
    generator = generator.cuda()
    sampler = get_sampler(H, embedding_weight, latent_loader).cuda()

    if H.deepspeed:
        model_engine, optim, _, _ = deepspeed.initialize(args=H, model=sampler, model_parameters=sampler.parameters())
    else:
        optim = torch.optim.Adam(sampler.parameters(), lr=H.lr)

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler =  copy.deepcopy(sampler)

    if H.load_step > 0:
        sampler =  load_model(sampler, H.sampler, H.load_step, H.load_dir).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_sampler =  load_model(ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
            except:
                ema_sampler =  copy.deepcopy(sampler)
        if H.load_optim:
            optim = load_model(optim, f'{H.sampler}_optim', H.load_step, H.load_dir)
            # only used when changing learning rates and reloading from checkpoint
            for param_group in optim.param_groups:
                param_group['lr'] = H.lr         

    losses = np.array([])
    mean_losses = np.array([])
    scaler = torch.cuda.amp.GradScaler()
    latent_iterator = cycle(latent_loader)

    start_step = H.load_step # defaults to 0 if not specified
    for step in range(start_step, H.train_steps):
        # lr warmup
        if H.warmup_iters:
            if step <= H.warmup_iters:
                optim_warmup(H, step, optim)

        x = next(latent_iterator)
        x = x.cuda()

        if H.deepspeed:
            optim.zero_grad()
            stats = sampler.train_iter(x) # don't need to cast to half() for diffusion?
            model_engine.backward(stats['loss'])
            model_engine.step()
        elif H.amp:
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                stats = sampler.train_iter(x)
            scaler.scale(stats['loss']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            stats = sampler.train_iter(x)
            
            if torch.isnan(stats['loss']).any():
                print(f'Skipping step {step} with NaN loss')
                continue
            optim.zero_grad()
            stats['loss'].backward()
            optim.step()

        losses = np.append(losses, stats['loss'].item())

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
            ema.update_model_average(ema_sampler, sampler)

        images = None
        if step % H.steps_per_display_output == 0 and step > 0:
            images = get_samples(H, generator, ema_sampler if H.ema else sampler)
            display_images(vis, images, H, win_name=f'{H.sampler}_samples')

        if step % H.steps_per_save_output == 0 and step > 0:
            if not images:
                images = get_samples(H, generator, ema_sampler if H.ema else sampler)
            save_images(images, 'samples', step, H.log_dir, H.save_individuallyH)

        if step % H.steps_per_checkpoint == 0 and step > H.load_step:
            save_model(sampler, H.sampler, step, H.log_dir)
            if H.ema:
                save_model(ema_sampler, f'{H.sampler}_ema', step, H.log_dir)
            else:
                save_model(optim, f'{H.sampler}_optim', step, H.log_dir)
            if H.sampler == 'ebm':
                save_buffer(sampler.buffer, step, H.log_dir)


if __name__=='__main__':
    H = get_sampler_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')   
    start_training_log(H)
    main(H, vis)