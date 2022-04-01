from torchvision.models import resnet18

from .lenet import *
from .resnet import *


class Hook:
    
    def __init__(self, config, name, module, previous_activations) -> None:
        self.name = name
        self.module = module
        self.samples_activation = []
        self.previous_activations = previous_activations
        self.activation_deltas = 0
        self.total_samples = 0
        
        self.config = config
        
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        
        # TODO sort this mess
        
        if self.config.mask_mode == "per-sample":
            reshaped_output = output.view(output.shape[0], output.shape[1], -1).mean(dim=2)
        if self.config.mask_mode == "per-feature":
            reshaped_output = output.view(output.shape[0], output.shape[1], -1)
        
        if self.previous_activations is None:
            self.samples_activation.append(reshaped_output)
        else:
            if self.config.delta_mode == "difference":
                delta = reshaped_output.float() \
                        - self.previous_activations[self.total_samples:output.shape[0] + self.total_samples].float()
            if self.config.delta_mode == "cosine":
                delta = 1 - torch.abs(
                    cosine_similarity(
                        reshaped_output.float(),
                        self.previous_activations[self.total_samples:output.shape[0] + self.total_samples].float(),
                        dim=0 if self.config.mask_mode == "per-sample" else 2))
            
            if self.config.mask_mode == "per-feature" and self.config.reduction == "mean":
                delta = torch.sum(delta, dim=0)
            
            if self.config.reduction == "mean":
                self.activation_deltas += delta
            else:
                if isinstance(self.activation_deltas, int):
                    self.activation_deltas = torch.max(delta, dim=0)[0]
                else:
                    self.activation_deltas = torch.maximum(torch.max(delta, dim=0)[0], self.activation_deltas)
            
            self.previous_activations[self.total_samples:output.shape[0] + self.total_samples] = reshaped_output
            self.total_samples += output.shape[0]
    
    def get_samples_activation(self):
        return torch.cat(self.samples_activation)
    
    def get_reduced_activation_delta(self):
        if self.config.mask_mode == "per-sample":
            if self.config.reduction == "mean":
                reduced_activation_delta = self.activation_deltas / self.total_samples
            elif self.config.reduction == "max":
                reduced_activation_delta = self.activation_deltas
        elif self.config.mask_mode == "per-feature":
            if self.config.delta_mode == "difference":
                if self.config.reduction == "mean":
                    reduced_activation_delta = torch.mean(self.activation_deltas / self.total_samples, dim=1)
                elif self.config.reduction == "max":
                    reduced_activation_delta = self.activation_deltas
            elif self.config.delta_mode == "cosine":
                if self.config.reduction == "mean":
                    reduced_activation_delta = self.activation_deltas / self.total_samples
                elif self.config.reduction == "max":
                    reduced_activation_delta = self.activation_deltas
        
        return reduced_activation_delta
    
    def reset(self):
        self.samples_activation = []
    
    def close(self) -> None:
        self.hook.remove()


def cosine_similarity(x1, x2, dim, eps=1e-8):
    x1_squared_norm = torch.pow(x1, 2).sum(dim=dim, keepdim=True)
    x2_squared_norm = torch.pow(x2, 2).sum(dim=dim, keepdim=True)
    
    # x1_squared_norm.clamp_min_(eps)
    # x2_squared_norm.clamp_min_(eps)
    
    x1_norm = x1_squared_norm.sqrt_()
    x2_norm = x2_squared_norm.sqrt_()
    
    x1_normalized = x1.div(x1_norm).nan_to_num(nan=0, posinf=0, neginf=0)
    x2_normalized = x2.div(x2_norm).nan_to_num(nan=0, posinf=0, neginf=0)
    
    mask_1 = (torch.abs(x1_normalized).sum(dim=dim) <= eps) * (torch.abs(x2_normalized).sum(dim=dim) <= eps)
    mask_2 = (torch.abs(x1_normalized).sum(dim=dim) > eps) * (torch.abs(x2_normalized).sum(dim=dim) > eps)
    
    cos_sim_value = torch.sum(x1_normalized * x2_normalized, dim=dim)
    
    return mask_2 * cos_sim_value + mask_1


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


def attach_hooks(config, model, hooks, previous_activations=None):
    for n, m in model.named_modules():
        # if n in arch_config["targets"]:
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            hooks[n] = Hook(config, n, m, previous_activations[n] if previous_activations is not None else None)
