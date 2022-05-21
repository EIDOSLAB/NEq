from torchvision.models import resnet18

from .lenet import *
from .resnet import *


def get_model(config):
    print(f'=> Initializing dataset {config.model}')
    
    model_dict = {"resnet32-cifar10":  resnet32,
                  "resnet18-imagenet": resnet18}
    
    model = model_dict[config.model]
    
    total_neurons = 0
    
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
            total_neurons += m.weight.shape[0]
    
    return model, total_neurons
