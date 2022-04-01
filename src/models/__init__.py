from torchvision.models import resnet18

from utils import cosine_similarity
from .lenet import *
from .resnet import *


class Hook:
    
    def __init__(self, config, name, module, pre_epoch_activations) -> None:
        self.name = name
        self.module = module
        self.samples_activation = []
        self.pre_epoch_activations = pre_epoch_activations
        self.activation_deltas = 0
        self.total_samples = 0
        
        self.config = config
        
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        if self.pre_epoch_activations is None:
            if self.config.mask_mode == "per-sample":
                # self.samples_activation.append(output.mean(dim=(2, 3)).mean(dim=0, keepdim=True))
                self.samples_activation.append(output.mean(dim=(2, 3)))
            if self.config.mask_mode == "per-feature":
                # self.samples_activation.append(output.mean(dim=0, keepdim=True))
                self.samples_activation.append(output.view(output.shape[0], output.shape[1], -1))
        
        if self.pre_epoch_activations is not None:
            # TODO all other cases
            self.activation_deltas += (1 - torch.abs(
                cosine_similarity(output.view(output.shape[0], output.shape[1], -1).float(),
                                  self.pre_epoch_activations[self.total_samples:output.shape[0] + self.total_samples].float(), dim=2)
            )).sum(dim=0)
            self.pre_epoch_activations[self.total_samples:output.shape[0] + self.total_samples] = output.view(output.shape[0],
                                                                                                      output.shape[1],
                                                                                                      -1)
            self.total_samples += output.shape[0]
    
    def get_samples_activation(self):
        return torch.cat(self.samples_activation)
    
    def get_reduced_activation_delta(self):
        return self.activation_deltas / self.total_samples
    
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


def attach_hooks(config, model, hooks, pre_epoch_activations=None):
    for n, m in model.named_modules():
        # if n in arch_config["targets"]:
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            hooks[n] = Hook(config, n, m, pre_epoch_activations[n] if pre_epoch_activations is not None else None)
