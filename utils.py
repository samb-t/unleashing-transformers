import torch
import torchvision
import logging
import numpy as np
import os

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def config_log(log_dir, dataset):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, f'log_{dataset}.txt'), encoding='utf-8', level=logging.DEBUG)

def log(output):
    print(output)
    logging.info(output)

def start_training_log(hparams):
    log('----------------')
    log('Starting training using following hparams:')
    for param in hparams:
        log(f'\t{param}: {hparams[param]:.4f}')

def save_model(model, model_save_name, step, log_dir):
    model_name = f'{model_save_name}_{step}.th'
    log(f'Saving model at step: {step}')
    torch.save(model.state_dict(), os.path.join(log_dir, model_name))


def load_model(model, model_load_name, step, log_dir):
    model.load_state_dict(torch.load(os.path.join(log_dir, f'{model_load_name}_{step}.th')))
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