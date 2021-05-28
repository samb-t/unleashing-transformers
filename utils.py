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
    logging.basicConfig(filename=os.path.join(log_dir, f'log.txt'), level=logging.DEBUG)


def log(output):
    print(output)
    logging.info(output)


def start_training_log(hparams):
    log('----------------')
    log('Starting training using following hparams:')
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

def get_data_loader(dataset_name, img_size, batch_size, num_workers=1, train=True, drop_last=True, download=False):
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST('~/Repos/_datasets', train=train, download=download, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size), torchvision.transforms.ToTensor()
        ]))
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('~/Repos/_datasets/CIFAR10', train=train, download=download, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size), torchvision.transforms.ToTensor()
    ]))
    elif dataset_name == 'flowers':
        dataset = torchvision.datasets.ImageFolder('~/Repos/_datasets/flowers', transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size), torchvision.transforms.CenterCrop(img_size), torchvision.transforms.ToTensor()
    ]))
    elif dataset_name =='celeba':
        dataset = torchvision.datasets.CelebA('~/Repos/_datasets/celeba', download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor()
        ]))
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, sampler=None, shuffle=True, batch_size=batch_size, drop_last=drop_last)
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


def update_model(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p2.data = p1.data