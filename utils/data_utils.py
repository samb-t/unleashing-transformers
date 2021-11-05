import torch
import torchvision
from torch.utils.data.dataset import Subset
from torchvision.transforms import Compose, Resize, CenterCrop, RandomHorizontalFlip, ToTensor


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def val_train_split(dataset, train_val_ratio):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_data_loader(
    dataset_name,
    img_size,
    batch_size,
    num_workers=1,
    drop_last=True,
    shuffle=True,
    get_val_dataset=True,
    get_flipped=False,
    train_val_split_ratio=0.95,
    dataset_path=None,
):

    transform = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])
    transform_with_flip = Compose([Resize(img_size), CenterCrop(img_size), RandomHorizontalFlip(p=1.0), ToTensor()])

    if dataset_name == 'churches':
        train_dataset = torchvision.datasets.LSUN(
            '/home2/kmhf27/workspace/data/LSUN',
            classes=['church_outdoor_train'],
            transform=transform
        )
        if get_flipped:
            train_dataset_flip = torchvision.datasets.LSUN(
                '/home2/kmhf27/workspace/data/LSUN',
                classes=['church_outdoor_train'],
                transform=transform_with_flip,
            )
        if get_val_dataset:
            val_dataset = torchvision.datasets.LSUN(
                '/home2/kmhf27/workspace/data/LSUN',
                classes=['church_outdoor_val'],
                transform=transform
            )

    elif dataset_name == 'bedrooms':
        train_dataset = torchvision.datasets.LSUN(
            '/projects/cgw/lsun',
            classes=['bedroom_train'],
            transform=transform,
        )
        if get_val_dataset:
            val_dataset = torchvision.datasets.LSUN(
                '/projects/cgw/lsun',
                classes=['bedroom_val'],
                transform=transform,
            )

        if get_flipped:
            train_dataset_flip = torchvision.datasets.LSUN(
                '/projects/cgw/lsun',
                classes=['bedroom_train'],
                transform=transform_with_flip,
            )

    elif dataset_name == 'ffhq':
        train_dataset = torchvision.datasets.ImageFolder(
            '/projects/cgw/FFHQ',
            transform=transform,
        )

        if get_flipped:
            train_dataset_flipped = torchvision.datasets.ImageFolder(
                '/projects/cgw/FFHQ',
                transform=transform_with_flip,
            )

        if get_val_dataset:
            train_dataset, val_dataset = val_train_split(train_dataset, train_val_split_ratio)
            if get_flipped:
                train_dataset_flipped, _ = val_train_split(train_dataset_flipped, train_val_split_ratio)

        if get_flipped:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_flipped])

    if get_flipped:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_flip])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        sampler=None,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last
    )
    if get_val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, num_workers=num_workers,
            sampler=None,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last
        )
    else:
        val_loader = None

    return train_loader, val_loader
