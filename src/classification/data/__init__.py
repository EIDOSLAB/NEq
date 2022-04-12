import torch.utils.data
import torchvision

from classification.data.transforms import T
from general_utils import split_dataset, MapDataset


def get_data(config):
    print(f"Initialize dataset {config.dataset}")
    
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
    
    print(f"Train set length {len(train)}")
    print(f"Validation set length {len(validation)}")
    print(f"Test set length {len(test_dataset)}")
    
    return train_dataloader, valid_dataloader, test_dataloader
