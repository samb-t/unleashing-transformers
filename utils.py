import torch
import torchvision
import logging
import numpy as np
import os

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def config_log(log_dir):
    log_dir = 'logs/' + log_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, f'log.txt'), encoding='utf-8', level=logging.DEBUG)


def log(output):
    print(output)
    logging.info(output)


def start_training_log(hparams):
    log('----------------')
    log('Starting training using following hparams:')
    for param in hparams:
        log(f'> {param}: {hparams[param]}')


def save_model(model, model_save_name, step, log_dir):
    log_dir = 'logs/' + log_dir + '/saved_models'
    os.makedirs(log_dir, exist_ok=True)
    model_name = f'{model_save_name}_{step}.th'
    log(f'Saving {model_save_name} at step {step}')
    torch.save(model.state_dict(), os.path.join(log_dir, model_name))


def load_model(model, model_load_name, step, log_dir):
    log_dir = 'logs/' + log_dir + '/saved_models'
    model.load_state_dict(torch.load(os.path.join(log_dir, f'{model_load_name}_{step}.th')))
    log(f'Loading {model_load_name} from step {step}')
    return model


def get_data_loader(dataset_name, img_size, batch_size, train=True, drop_last=True):
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST('~/Repos/_datasets', train=train, download=False, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size), torchvision.transforms.ToTensor()
        ]))
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('~/Repos/_datasets/CIFAR10', train=train, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size), torchvision.transforms.ToTensor()
    ]))
    
    loader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, batch_size=batch_size, drop_last=drop_last)
    return loader


def save_images(images, vis_win, win_name, step, log_dir):
    log_dir = 'logs/' + log_dir + '/images'
    os.makedirs(log_dir, exist_ok=True)
    images = torch.clamp(images, 0, 1)
    vis_win.images(torch.clamp(images, 0, 1), nrow=int(np.sqrt(images.shape[0])), win=win_name, opts=dict(title=win_name))
    torchvision.utils.save_image(torch.clamp(images, 0, 1), f'{log_dir}/{win_name}_{step}.png', 
            nrow=int(np.sqrt(images.shape[0])), padding=0)

def save_latents(latents, dataset):
    save_dir = 'latents/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(latents, f'latents/{dataset}_latents.pkl')