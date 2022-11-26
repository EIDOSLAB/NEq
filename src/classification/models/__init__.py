from torchvision.models import resnet18

from .resnet import *


def get_model(config):
    print(f"Initialize model {config.arch}")
    
    if config.arch == "resnet32-cifar":
        model = resnet32()
    elif config.arch == "resnet18-imagenet":
        model = resnet18(False)
    else:
        raise ValueError(f"No such model {config.arch}")
    
    total_neurons = 0
    
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total_neurons += m.weight.shape[0]
        if isinstance(m, nn.Conv2d):
            total_neurons += m.weight.shape[0]
        if isinstance(m, nn.BatchNorm2d):
            total_neurons += m.weight.shape[0]
    
    return model, total_neurons
