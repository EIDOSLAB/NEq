import yaml
from yaml import FullLoader

from .lenet import *
from .resnet import *


class Hook:
    
    def __init__(self, config, name, module) -> None:
        self.name = name
        self.module = module
        self.samples_activation = []
        self.active_count = torch.zeros(module.weight.shape[0], device=config.device)
        
        self.config = config
        
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        if self.config.mask_mode == "per-sample":
            self.samples_activation.append(output.mean(dim=(2, 3)))
        if self.config.mask_mode == "per-feature":
            self.samples_activation.append(output)
    
    def get_samples_activation(self):
        return torch.cat(self.samples_activation)
    
    def reset(self):
        self.samples_activation = []
    
    def close(self) -> None:
        self.hook.remove()


def get_model(config):
    if config.arch == "lenet5":
        model = LeNet5()
    elif config.arch == "resnet32-cifar":
        model = resnet32()
    else:
        raise ValueError(f"No such model {config.arch}")
    
    with open(f'models/configs/{config.arch}.yaml') as f:
        arch_config = yaml.load(f.read(), Loader=FullLoader)
    
    return model, arch_config


def attach_hooks(config, model, hooks, arch_config):
    for n, m in model.named_modules():
        if n in arch_config["targets"]:
            hooks[n] = Hook(config, n, m)
