from torchvision.models import resnet18

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
    print(f"Initialize model {config.arch}")
    
    if config.arch == "lenet5":
        model = LeNet5()
    elif config.arch == "resnet32-cifar":
        model = resnet32()
    elif config.arch == "resnet18-imagenet":
        model = resnet18(False)
    else:
        raise ValueError(f"No such model {config.arch}")
    
    # with open(f'models/configs/{config.arch}.yaml') as f:
    #     arch_config = yaml.load(f.read(), Loader=FullLoader)
    
    total_neurons = 0
    
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total_neurons += m.weight.shape[0]
        if isinstance(m, nn.Conv2d):
            total_neurons += m.weight.shape[0]
        if isinstance(m, nn.BatchNorm2d):
            total_neurons += m.weight.shape[0]
    
    return model, None, total_neurons


def attach_hooks(config, model, hooks, arch_config):
    for n, m in model.named_modules():
        # if n in arch_config["targets"]:
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            hooks[n] = Hook(config, n, m)
