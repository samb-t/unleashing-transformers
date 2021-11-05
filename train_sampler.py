import torch
import deepspeed
import numpy as np
import copy
import time
import os
from tqdm import tqdm
from models import \
    VQAutoEncoder, Generator,\
    AbsorbingDiffusion, Transformer, AutoregressiveTransformer
from hparams import get_sampler_hparams
from utils.data_utils import get_data_loaders, cycle
from utils.sampler_utils import generate_latent_ids, get_latent_loaders, retrieve_autoencoder_components_state_dicts,\
    get_samples, unpack_sampler_stats
from utils.train_utils import EMA, optim_warmup
from utils.log_utils import log, log_stats, setup_visdom, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images, \
    display_images

# torch.backends.cudnn.benchmark = True


def get_sampler(H, embedding_weight):

    if H.sampler == 'absorbing':
        denoise_fn = Transformer(H).cuda()
        sampler = AbsorbingDiffusion(
            H, denoise_fn, H.codebook_size, embedding_weight)

    elif H.sampler == 'autoregressive':
        sampler = AutoregressiveTransformer(H, embedding_weight)

    return sampler


def main(H, vis):

    latents_fp_suffix = '_flipped' if H.horizontal_flip else ''
    latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_train_latents{latents_fp_suffix}'

    print(latents_filepath)

    train_with_validation_dataset = False
    if H.steps_per_eval:
        train_with_validation_dataset = True

    if not os.path.exists(latents_filepath):
        ae_state_dict = retrieve_autoencoder_components_state_dicts(
            H, ['encoder', 'quantize', 'generator']
        )
        ae = VQAutoEncoder(H)
        ae.load_state_dict(ae_state_dict)
        # val_loader will be assigned to None if not training with validation dataest
        train_loader, val_loader = get_data_loaders(
            H.dataset,
            H.img_size,
            H.batch_size,
            drop_last=False,
            shuffle=False,
            get_flipped=H.horizontal_flip,
            get_val_dataloader=train_with_validation_dataset
        )

        log("Transferring autoencoder to GPU to generate latents...")
        ae = ae.cuda()  # put ae on GPU for generating
        generate_latent_ids(H, ae, train_loader, val_loader)
        log("Deleting autoencoder to conserve GPU memory...")
        ae = ae.cpu()
        ae = None

    train_latent_loader, val_latent_loader = get_latent_loaders(H, get_validation_loader=train_with_validation_dataset)

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )

    embedding_weight = quanitzer_and_generator_state_dict.pop(
        'embedding.weight')
    if H.deepspeed:
        embedding_weight = embedding_weight.half()
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)

    # NOTE: can move generator to cpu to save memory if needbe - add flag?
    generator.load_state_dict(quanitzer_and_generator_state_dict)
    generator = generator.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()

    if H.deepspeed:
        model_engine, optim, _, _ = deepspeed.initialize(
            args=H, model=sampler, model_parameters=sampler.parameters())
    else:
        optim = torch.optim.Adam(sampler.parameters(), lr=H.lr)

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    # initialise before loading so as not to overwrite loaded stats
    losses = np.array([])
    val_losses = np.array([])
    elbo = np.array([])
    mean_losses = np.array([])
    start_step = 0
    log_start_step = 0
    if H.load_step > 0:
        start_step = H.load_step + 1

        sampler = load_model(sampler, H.sampler,
                             H.load_step, H.load_dir).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
            except:
                ema_sampler = copy.deepcopy(sampler)
        if H.load_optim:
            optim = load_model(
                optim, f'{H.sampler}_optim', H.load_step, H.load_dir)
            # only used when changing learning rates and reloading from checkpoint
            for param_group in optim.param_groups:
                param_group['lr'] = H.lr

        try:
            train_stats = load_stats(H, H.load_step)
        except:
            train_stats = None

        if train_stats is not None:
            losses, mean_losses, val_losses, elbo, H.steps_per_log = unpack_sampler_stats(
                train_stats)
            log_start_step = 0

            # initialise plots
            vis.line(mean_losses, list(range(log_start_step, start_step, H.steps_per_log)),
                    win='loss', opts=dict(title='Loss'))
            vis.line(elbo, list(range(log_start_step, start_step, H.steps_per_log)),
                    win='ELBO', opts=dict(title='ELBO'))
            vis.line(val_losses, list(range(H.steps_per_eval, start_step, H.steps_per_eval)),
                    win='Val_loss', opts=dict(title='Validation Loss'))
        else:
            log('No stats file found for loaded model, displaying stats from load step only.')
            log_start_step = start_step

    scaler = torch.cuda.amp.GradScaler()
    train_iterator = cycle(train_latent_loader)
    val_iterator = cycle(val_latent_loader)

    print("params", sum(p.numel() for p in sampler.parameters()))

    for step in range(start_step, H.train_steps):
        step_start_time = time.time()
        # lr warmup
        if H.warmup_iters:
            if step <= H.warmup_iters:
                optim_warmup(H, step, optim)

        x = next(train_iterator)
        x = x.cuda()

        if H.deepspeed:
            optim.zero_grad()
            stats = sampler.train_iter(x)
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
            step_time_taken = time.time() - step_start_time
            stats['step_time'] = step_time_taken
            mean_loss = np.mean(losses)
            stats['mean_loss'] = mean_loss
            mean_losses = np.append(mean_losses, mean_loss)
            losses = np.array([])

            vis.line(np.array([mean_loss]), np.array([step]), win='loss',
                     update=('append' if step > 0 else 'replace'), opts=dict(title='Loss'))
            log_stats(step, stats)

            if H.sampler == 'absorbing':
                elbo = np.append(elbo, stats['vb_loss'].item())
                vis.bar(sampler.loss_history, list(range(sampler.loss_history.size(0))),
                        win='loss_bar', opts=dict(title='loss_bar'))

                vis.line(np.array([stats['vb_loss'].item()]), np.array([step]),
                         win='ELBO', update=('append' if step > 0 else 'replace'), opts=dict(title='ELBO'))

        if H.ema and step % H.steps_per_update_ema == 0 and step > 0:
            ema.update_model_average(ema_sampler, sampler)

        images = None
        if step % H.steps_per_display_output == 0 and step > 0:
            images = get_samples(
                H, generator, ema_sampler if H.ema else sampler)
            display_images(vis, images, H, win_name=f'{H.sampler}_samples')

        if step % H.steps_per_save_output == 0 and step > 0:
            if images == None:
                images = get_samples(
                    H, generator, ema_sampler if H.ema else sampler)
            save_images(images, 'samples', step,
                        H.log_dir, H.save_individually)

        if H.steps_per_eval and step % H.steps_per_eval == 0 and step > 0:
            # calculate validation loss
            valid_loss, valid_elbo, num_samples = 0.0, 0.0, 0
            eval_repeats = 3
            print("Evaluating")
            for _ in tqdm(range(eval_repeats)):
                x = next(val_iterator)
                with torch.no_grad():
                    stats = sampler.train_iter(x.cuda())
                    valid_loss += stats['loss'].item()
                    if H.sampler == 'absorbing':
                        valid_elbo += stats['vb_loss'].item()
                    num_samples += x.size(0)
            valid_loss = valid_loss / num_samples
            if H.sampler == 'absorbing':
                valid_elbo = valid_elbo / num_samples

            val_losses = np.append(val_losses, valid_loss)
            vis.line(np.array([valid_loss]), np.array([step]),
                    win='Val_loss', update=('append' if step > 0 else 'replace'), opts=dict(title='Validation Loss'))

            if H.sampler == 'absorbing':
                vis.line(np.array([valid_elbo]), np.array([step]),
                        win='Val_elbo', update=('append' if step > 0 else 'replace'), opts=dict(title='Validation ELBO'))

        if step % H.steps_per_checkpoint == 0 and step > H.load_step:
            save_model(sampler, H.sampler, step, H.log_dir)
            save_model(optim, f'{H.sampler}_optim', step, H.log_dir)

            if H.ema:
                save_model(ema_sampler, f'{H.sampler}_ema', step, H.log_dir)


            train_stats = {
                'losses': losses,
                'mean_losses': mean_losses,
                'val_losses': val_losses,
                'elbo': elbo,
                'steps_per_log': H.steps_per_log
            }
            save_stats(H, train_stats, step)


if __name__ == '__main__':
    H = get_sampler_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)
    main(H, vis)
