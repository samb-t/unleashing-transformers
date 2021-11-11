import imageio
import os
import torch
import torchvision
from torch.utils.data.dataset import Subset
from torchvision.transforms import Compose, Resize, CenterCrop, RandomHorizontalFlip, ToTensor


class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_paths = os.listdir(folder)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = imageio.imread(self.folder+path)
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> channels first
        return img

    def __len__(self):
        return len(self.image_paths)


class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        self.length = length if length is not None else len(dataset)

    def __getitem__(self, index):
        return self.dataset[index][0].mul(255).clamp_(0, 255).to(torch.uint8)

    def __len__(self):
        return self.length


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train_val_split(dataset, train_val_ratio):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_datasets(
    dataset_name,
    img_size,
    get_val_dataset=False,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
):
    transform = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])
    transform_with_flip = Compose([Resize(img_size), CenterCrop(img_size), RandomHorizontalFlip(p=1.0), ToTensor()])

    if dataset_name in ["churches", "bedrooms"]:
        dataset_path = "/projects/cgw/lsun"
    elif dataset_name == "ffhq":
        dataset_path = "/projects/cgw/FFHQ"
    elif dataset_name == "custom":
        if custom_dataset_path:
            dataset_path = custom_dataset_path
        else:
            raise ValueError("Custom dataset selected, but no path provided")
    else:
        raise ValueError(f"Invalid dataset chosen: {dataset_name}. To use a custom dataset, set --dataset \
            flag to 'custom'.")

    if dataset_name == "churches":
        train_dataset = torchvision.datasets.LSUN(
            dataset_path,
            classes=["church_outdoor_train"],
            transform=transform
        )
        if get_flipped:
            train_dataset_flip = torchvision.datasets.LSUN(
                dataset_path,
                classes=["church_outdoor_train"],
                transform=transform_with_flip,
            )
        if get_val_dataset:
            val_dataset = torchvision.datasets.LSUN(
                dataset_path,
                classes=["church_outdoor_val"],
                transform=transform
            )

    elif dataset_name == "bedrooms":
        train_dataset = torchvision.datasets.LSUN(
            dataset_path,
            classes=["bedroom_train"],
            transform=transform,
        )
        if get_val_dataset:
            val_dataset = torchvision.datasets.LSUN(
                dataset_path,
                classes=["bedroom_val"],
                transform=transform,
            )

        if get_flipped:
            train_dataset_flip = torchvision.datasets.LSUN(
                dataset_path,
                classes=["bedroom_train"],
                transform=transform_with_flip,
            )

    elif dataset_name == "ffhq" or "custom":
        train_dataset = torchvision.datasets.ImageFolder(
            dataset_path,
            transform=transform,
        )

        if get_flipped:
            train_dataset_flip = torchvision.datasets.ImageFolder(
                dataset_path,
                transform=transform_with_flip,
            )

        if get_val_dataset:
            train_dataset, val_dataset = train_val_split(train_dataset, train_val_split_ratio)
            if get_flipped:
                train_dataset_flip, _ = train_val_split(train_dataset_flip, train_val_split_ratio)

    if get_flipped:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_flip])

    if not get_val_dataset:
        val_dataset = None

    return train_dataset, val_dataset


def get_data_loaders(
    dataset_name,
    img_size,
    batch_size,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
    num_workers=1,
    drop_last=True,
    shuffle=True,
    get_val_dataloader=False,
):

    train_dataset, val_dataset = get_datasets(
        dataset_name,
        img_size,
        get_flipped=get_flipped,
        get_val_dataset=get_val_dataloader,
        train_val_split_ratio=train_val_split_ratio,
        custom_dataset_path=custom_dataset_path,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        sampler=None,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last
    )
    if get_val_dataloader:
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
