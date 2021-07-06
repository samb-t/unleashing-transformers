import torch
import torchvision
import numpy as np
import logging
import os

def config_log(log_dir):
    log_dir = 'logs/' + log_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, f'log.txt'), level=logging.INFO)


def log(output):
    print(output)
    logging.info(output)

def log_stats(step, stats):
    log_str = f'Step: {step}  '
    for stat in stats:
        if stat != 'images':
            log_str += f'{stat}: {stats[stat]:.4f}  '
    log(log_str)


def start_training_log(hparams):
    log(f"Using following hparams:")
    for param in hparams:
        log(f'> {param}: {hparams[param]}')


def save_model(model, model_save_name, suffix, log_dir):
    log_dir = 'logs/' + log_dir + '/saved_models'
    os.makedirs(log_dir, exist_ok=True)
    model_name = f'{model_save_name}_{suffix}.th'
    log(f'Saving {model_save_name} to {model_save_name}_{str(suffix)}.th')
    torch.save(model.state_dict(), os.path.join(log_dir, model_name))


def load_model(model, model_load_name, suffix, log_dir):
    log_dir = 'logs/' + log_dir + '/saved_models'
    model.load_state_dict(torch.load(os.path.join(log_dir, f'{model_load_name}_{suffix}.th')))
    log(f'Loading {model_load_name}_{str(suffix)}.th')
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
    vis.images(torch.clamp(images, 0, 1), win=win_name, opts=dict(title=win_name))


def save_images(images, im_name, step, log_dir):
    log_dir = 'logs/' + log_dir + '/images'
    os.makedirs(log_dir, exist_ok=True)
    torchvision.utils.save_image(torch.clamp(images, 0, 1), f'{log_dir}/{im_name}_{step}.png', 
            nrow=int(np.sqrt(images.shape[0])), padding=0)


def save_latents(latents, dataset, size):
    save_dir = 'latents/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(latents, f'latents/{dataset}_{size}_latents')


def save_latents(latents, dataset, size):
    save_dir = 'latents/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(latents, f'latents/{dataset}_{size}_latents')