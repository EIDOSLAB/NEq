from typing import Tuple

import numpy as np
import torch.utils.data
import torchvision
from torch import Generator
from torch.utils.data import random_split, Dataset

from data.transforms import T


class MapDataset(Dataset):
    """Given a dataset, creates a dataset which applies a mapping function to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.

    Args:
        dataset:
        map_fn:
    """
    
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn
    
    def __getitem__(self, index):
        return self.map(self.dataset[index][0]), self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)


def split_dataset(dataset: torch.utils.data.Dataset, percentage: float, random_seed: int = 0) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Randomly splits a `torch.utils.data.Dataset` instance in two non-overlapping separated `Datasets`.

    The split of the elements of the original `Dataset` is based on `percentage` $$\in [0, 1]$$.
    I.e. if `percentage=0.2` the first returned dataset will contain 80% of the total elements and the second 20%.

    Args:
        dataset (torch.utils.data.Dataset): `torch.utils.data.Dataset` instance to be split.
        percentage (float): percentage of elements of `dataset` contained in the second dataset.
        random_seed (int): random seed for the split generator.

    Returns:
        tuple: a tuple containing the two new datasets.

    """
    dataset_length = len(dataset)
    valid_length = int(np.floor(percentage * dataset_length))
    train_length = dataset_length - valid_length
    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length],
                                                generator=Generator().manual_seed(random_seed))
    
    return train_dataset, valid_dataset


def get_data(config):
    if config.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(config.root, train=True, transform=None, download=True)
        train, validation = split_dataset(train_dataset, config.val_size)
        
        train, validation = MapDataset(train, T["cifar10"][0]), MapDataset(validation, T["cifar10"][1])
        
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True,
                                                       num_workers=config.num_workers, pin_memory=True,
                                                       persistent_workers=config.num_workers > 0)
        
        valid_dataloader = torch.utils.data.DataLoader(validation, batch_size=config.batch_size, shuffle=False,
                                                       num_workers=config.num_workers, pin_memory=True,
                                                       persistent_workers=config.num_workers > 0)
        
        test_dataset = torchvision.datasets.CIFAR10(config.root, train=False, transform=T["cifar10"][1], download=True)
        
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                      num_workers=config.num_workers, pin_memory=True,
                                                      persistent_workers=config.num_workers > 0)
    elif config.dataset == "imagenet":
        train_dataset = torchvision.datasets.ImageNet(config.root, split="train", transform=None)
        train, validation = split_dataset(train_dataset, config.val_size)
        
        train, validation = MapDataset(train, T["imagenet"][0]), MapDataset(validation, T["imagenet"][1])
        
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True,
                                                       num_workers=config.num_workers, pin_memory=True,
                                                       persistent_workers=config.num_workers > 0)
        
        valid_dataloader = torch.utils.data.DataLoader(validation, batch_size=config.batch_size, shuffle=False,
                                                       num_workers=config.num_workers, pin_memory=True,
                                                       persistent_workers=config.num_workers > 0)
        
        test_dataset = torchvision.datasets.ImageNet(config.root, split="val", transform=T["imagenet"][1])
        
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                      num_workers=config.num_workers, pin_memory=True,
                                                      persistent_workers=config.num_workers > 0)
    
    else:
        raise ValueError(f"No such dataset {config.dataset}")
    
    return train_dataloader, valid_dataloader, test_dataloader
