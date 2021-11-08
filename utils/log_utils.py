import torch
import torchvision
import numpy as np
import logging
import os
import visdom


def config_log(log_dir, filename='log.txt'):
    log_dir = 'logs/' + log_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, filename), 
        level=logging.INFO,
        format='%(message)s'
    )


def log(output):
    print(output)
    logging.info(output)


def log_stats(step, stats):
    log_str = f'Step: {step}  '
    for stat in stats:
        if 'latent_ids' not in stat and 'x_hat' not in stat and 'x_aug' not in stat and 'x_hat_aug' not in stat:
            log_str += f'{stat}: {stats[stat]:.4f}  '
    log(log_str)


def start_training_log(hparams):
    log(f"Using following hparams:")
    for param in hparams:
        log(f'> {param}: {hparams[param]}')


def save_model(model, model_save_name, step, log_dir):
    log_dir = 'logs/' + log_dir + '/saved_models'
    os.makedirs(log_dir, exist_ok=True)
    model_name = f'{model_save_name}_{step}.th'
    log(f'Saving {model_save_name} to {model_save_name}_{str(step)}.th')
    torch.save(model.state_dict(), os.path.join(log_dir, model_name))


def load_model(model, model_load_name, step, log_dir):
    log_dir = 'logs/' + log_dir + '/saved_models'
    model.load_state_dict(torch.load(os.path.join(log_dir, f'{model_load_name}_{step}.th')))
    log(f'Loading {model_load_name}_{str(step)}.th')
    return model


def save_buffer(buffer, step, log_dir):
    log_dir = 'logs/' + log_dir + '/saved_models'
    torch.save(buffer, os.path.join(log_dir, f'buffer_{step}.pt'))


def load_buffer(name, log_dir):
    log_dir = 'logs/' + log_dir + '/saved_models'
    buffer = torch.load(os.path.join(log_dir, f'buffer_{name}.pt'))
    return buffer


def display_images(vis, images, H, win_name=None):
    if win_name == None:
        win_name = f'{H.model}_images'
    images = torchvision.utils.make_grid(images.clamp(0,1), nrow=int(np.sqrt(images.size(0))), padding=0)
    vis.image(images, win=win_name, opts=dict(title=win_name))


def save_images(images, im_name, step, log_dir, save_indivudally=False):
    log_dir = 'logs/' + log_dir + '/images'
    os.makedirs(log_dir, exist_ok=True)
    if save_indivudally:
        for idx in range(len(images)):
            torchvision.utils.save_image(torch.clamp(images[idx], 0, 1), f'{log_dir}/{im_name}_{step}_{idx}.png')
    else:
        torchvision.utils.save_image(torch.clamp(images, 0, 1), f'{log_dir}/{im_name}_{step}.png', 
                nrow=int(np.sqrt(images.shape[0])), padding=0)


def save_latents(latents, dataset, size):
    save_dir = 'latents/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(latents, f'latents/{dataset}_{size}_latents')


def save_latents(H, train_latent_ids, val_latent_ids):

    save_dir = 'latents/'
    os.makedirs(save_dir, exist_ok=True)
    if H.flip_images:
        train_latents_fp = f'latents/{H.dataset}_{H.latent_shape[-1]}_train_latents_flip{H.latent_append}'
        val_latents_fp = f'latents/{H.dataset}_{H.latent_shape[-1]}_val_latents_flip{H.latent_append}'
    else:
        train_latents_fp = f'latents/{H.dataset}_{H.latent_shape[-1]}_train_latents{H.latent_append}'
        val_latents_fp = f'latents/{H.dataset}_{H.latent_shape[-1]}_val_latents{H.latent_append}'
    
    torch.save(train_latent_ids, train_latents_fp)
    torch.save(val_latent_ids, val_latents_fp)


def save_stats(H, stats, step):
    save_dir = f'logs/{H.log_dir}/saved_stats'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'logs/{H.log_dir}/saved_stats/stats_{step}'
    log(f'Saving stats to {save_path}')
    torch.save(stats, save_path)


def load_stats(H, step):
    load_path = f'logs/{H.load_dir}/saved_stats/stats_{step}'
    stats = torch.load(load_path)
    return stats


def setup_visdom(H):
    if H.ncc:
        server = 'ncc1.clients.dur.ac.uk'
    else:
        server = None

    if server:
        vis = visdom.Visdom(server=server, port=H.visdom_port)
    else:
        vis = visdom.Visdom(port=H.visdom_port)
    return vis
