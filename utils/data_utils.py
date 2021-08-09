import torch
import torchvision


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_data_loader(dataset_name, img_size, batch_size, num_workers=1, train=True, drop_last=True, download=False, shuffle=True):
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST('~/Repos/_datasets', train=train, download=download, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ]))
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('~/Repos/_datasets/CIFAR10', train=train, download=download, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ]))
    elif dataset_name == 'flowers':
        dataset = torchvision.datasets.ImageFolder('~/Repos/_datasets/flowers', transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor()
        ]))
    elif dataset_name == 'churches':
        dataset = torchvision.datasets.LSUN('../../../data/LSUN', classes=['church_outdoor_train'], transform=torchvision.transforms.Compose([
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
    elif dataset_name == 'ffhq':
        dataset = torchvision.datasets.ImageFolder('~/Repos/_datasets/FFHQ',  transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ]))
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, sampler=None, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
    return loader


