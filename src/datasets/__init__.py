from filelock import FileLock
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageNet

from datasets.map import MapDataset
from datasets.split import split_dataset
from datasets.transforms import T


def get_dataloaders(config, shuffle=True):
    print(f'=> Loading dataset {config.dataset}')
    
    loader = {"cifar10":  load_cifar10,
              "imagenet": load_imagenet}
    
    return loader[config.dataset](config, shuffle=shuffle)


def load_cifar10(config, shuffle):
    transform = T["cifar10"]
    
    with FileLock("mnist.lock"):
        train_dataset = CIFAR10(root=config.root, train=True, transform=None, download=True)
        train_dataset, validation_dataset = split_dataset(train_dataset, config.validation_size, config.seed)
        test_dataset = CIFAR10(root=config.root, train=False, transform=transform[1], download=True)
    
    return _get_dataloaders(config, train_dataset, validation_dataset, test_dataset, transform, shuffle)


def load_imagenet(config, shuffle):
    transform = T["mnist"]
    
    with FileLock("mnist.lock"):
        train_dataset = ImageNet(root=config.root, split="train", transform=None)
        train_dataset, validation_dataset = split_dataset(train_dataset, config.validation_size, config.seed)
        test_dataset = ImageNet(root=config.root, split="val", transform=transform[1])
    
    return _get_dataloaders(config, train_dataset, validation_dataset, test_dataset, transform, shuffle)


def _get_dataloaders(config, train_dataset, validation_dataset, test_dataset, transform, shuffle):
    train_dataset = MapDataset(train_dataset, transform[0])
    validation_dataset = MapDataset(validation_dataset, transform[1])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=shuffle, num_workers=config.num_workers,
                              persistent_workers=config.num_workers > 0, pin_memory=True)
    
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=config.batch_size,
                                   shuffle=False, num_workers=config.num_workers,
                                   persistent_workers=config.num_workers > 0, pin_memory=True)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers,
                             persistent_workers=config.num_workers > 0, pin_memory=True)
    
    return {"train":      train_loader,
            "validation": validation_loader,
            "test":       test_loader}
