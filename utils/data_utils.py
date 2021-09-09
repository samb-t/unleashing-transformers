import torch
import torchvision
from torch.utils.data.dataset import Subset


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def val_train_split(dataset, train_ratio=0.9):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_data_loader(dataset_name, img_size, batch_size, num_workers=1, drop_last=True, download=False, shuffle=True, get_val_train_split=True):
    if dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST('~/Repos/_datasets', train=True, download=download, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ]))
        val_dataset = torchvision.datasets.MNIST('~/Repos/_datasets', train=False, download=download, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ]))
    elif dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10('~/Repos/_datasets/CIFAR10', train=True, download=download, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ]))
        val_dataset = torchvision.datasets.CIFAR10('~/Repos/_datasets/CIFAR10', train=False, download=download, transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(img_size),
                torchvision.transforms.ToTensor()
            ]))
    elif dataset_name == 'flowers':
        dataset = torchvision.datasets.ImageFolder('~/Repos/_datasets/flowers', transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor()
        ]))
        train_dataset, val_dataset = val_train_split(dataset)
    elif dataset_name == 'churches':
        train_dataset = torchvision.datasets.LSUN('/home/sam/workspace/data/LSUN', classes=['church_outdoor_train'], transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor()
        ]))
        val_dataset = torchvision.datasets.LSUN('/home/sam/workspace/data/LSUN', classes=['church_outdoor_val'], transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor()
        ]))
        
    elif dataset_name =='celeba':
        dataset = torchvision.datasets.CelebA('~/Repos/_datasets/celeba', download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor()
        ]))

        if get_val_train_split:
            train_dataset, val_dataset = val_train_split(dataset)
        else:
            train_dataset, val_dataset = dataset, None

    elif dataset_name == 'ffhq':
        dataset = torchvision.datasets.ImageFolder('~/Repos/_datasets/FFHQ',  transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ]))
        
        if get_val_train_split:
            train_dataset, val_dataset = val_train_split(dataset)
        else:
            train_dataset, val_dataset = dataset, None

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers, sampler=None, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
    if val_dataset != None:
        val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=num_workers, sampler=None, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
    else:
        val_loader = None
        
    return train_loader, val_loader


