import torch
import torch.nn as nn
import numpy as np
import math
import time
from models.vqgan import VQGAN
from tqdm import tqdm
from hparams import get_sampler_hparams
from utils.sampler_utils import get_sampler, retrieve_autoencoder_components_state_dicts
from utils.data_utils import get_data_loader, cycle
from utils.log_utils import (
    log, log_stats, save_model,
    display_images, setup_visdom,
    config_log, start_training_log,
    load_model
)

torch.backends.cudnn.benchmark = True


def main(H, vis):
    vqgan = VQGAN(H).cuda()

    train_loader, val_loader = get_data_loader(
        H.dataset,
        H.img_size,
        H.batch_size,
        get_val_train_split=(H.steps_per_eval != 0)
    )
    train_iterator = cycle(train_loader)

    vqgan = load_model(vqgan, 'vqgan_ema', H.load_step, H.load_dir)

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ["quantize", "generator"],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop(
        "embedding.weight")
    embedding_weight = embedding_weight.cuda()

    sampler = get_sampler(H, embedding_weight)
    if H.load_step > 0:
        sampler = load_model(
            sampler, f"{H.sampler}_ema", H.load_step, H.load_dir).cuda()

    sampler = sampler.eval()
    sampler.num_timesteps = 256

    optim = torch.optim.Adam(vqgan.ae.generator.logsigma.parameters(), lr=1e-4)

    if H.amp:
        scaler = torch.cuda.amp.GradScaler()

    losses = np.array([])
    mean_losses = np.array([])

    for step in range(H.train_steps):
        step_start_time = time.time()
        batch = next(train_iterator)
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch

        x = x.cuda()

        optim.zero_grad()
        if H.amp:
            with torch.cuda.amp.autocast():
                x_hat, stats = vqgan.probabilistic(x, step)
            scaler.scale(stats['nll']).backward()
            scaler.step(optim)
            scaler.update()
        else:
            x_hat, stats = vqgan.probabilistic(x, step)
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
                list(range(0, step+1, H.steps_per_log)),
                win='loss',
                opts=dict(title='Loss')
            )
            log_stats(step, stats)

        if step % H.steps_per_eval == 0 and step > 0:
            sampler = sampler.cuda()
            with torch.no_grad():
                bpds = []
                for x in tqdm(val_loader, total=len(val_loader)):
                    if isinstance(x, list):
                        x = x[0]
                    x = x.cuda()

                    x_hat, stats = vqgan.probabilistic(x)
                    nl_p_x_z = stats["nll_raw"]

                    z = stats["latent_ids"]
                    nl_p_z = sampler.elbo(z)

                    pixels = 256 * 256 * 3

                    nl_p_x = nl_p_x_z + nl_p_z + float(math.log(32.) * pixels)  # 5 bit
                    bpd = nl_p_x / (pixels * math.log(2.0))
                    bpds.extend(bpd.tolist())

            log(f"ELBO approximation: {torch.tensor(bpds).mean()}")
            sampler = sampler.cpu()

        if step % H.steps_per_display_output == 0 and step > 0:
            display_images(vis, x, H, 'Original Images')
            x_hat = x_hat.detach().cpu().to(torch.float32)
            display_images(vis, x_hat, H, 'VQGAN Recons')

        if step % H.steps_per_checkpoint == 0 and step > H.load_step:
            save_model(vqgan, 'vqgan_probabilistic', step, H.log_dir)
            save_model(optim, 'optim_probabilistic', step, H.log_dir)


if __name__ == '__main__':
    H = get_sampler_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for VQGAN on {H.dataset}')
    start_training_log(H)
    main(H, vis)
