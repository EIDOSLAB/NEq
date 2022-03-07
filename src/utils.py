import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from yaml import FullLoader

from models import LeNet5, resnet32


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
            self.samples_activation.append(F.relu(output).mean(dim=(2, 3)))
        if self.config.mask_mode == "per-feature":
            self.samples_activation.append(F.relu(output))
    
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


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def topk_accuracy(outputs, labels, topk=1):
    outputs = torch.softmax(outputs, dim=1)
    _, preds = outputs.topk(topk, dim=1)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds)).sum()
    return (correct / float(len(outputs))).cpu().item()


def run(config, model, dataloader, optimizer, scaler, device, grad_mask):
    train = optimizer is not None
    
    tot_loss = 0.
    outputs = []
    targets = []
    
    model.train(train)
    pbar = tqdm(dataloader, desc="Training" if train else "Testing",
                disable=(dist.is_initialized() and dist.get_rank() > 0))
    
    for images, target in pbar:
        images, target = images.to(device, non_blocking=True), \
                         target.to(device, non_blocking=True)
        
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(config.device == "cuda" and config.amp)):
                output = model(images)
                
                loss = F.cross_entropy(output, target)
                
                if train:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    
                    if config.rollback != "optim":
                        if config.rollback == "manual":
                            pre_optim_state = deepcopy(model.state_dict())
                        for k in grad_mask:
                            if config.rollback == "none":
                                zero_gradients(model, k, grad_mask[k])
                            if config.rollback == "manual":
                                rollback_module(model, k, grad_mask[k], pre_optim_state)
                    
                    scaler.step(optimizer)
                    scaler.update()
        
        tot_loss += loss.item()
        outputs.append(output.detach().float())
        targets.append(target)
    
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    
    accs = {
        'top1': topk_accuracy(outputs, targets, topk=1),
        'top5': topk_accuracy(outputs, targets, topk=5)
    }
    return {'loss': tot_loss / len(dataloader.dataset), 'accuracy': accs}


@torch.no_grad()
def zero_gradients(model, name, mask):
    module = find_module_by_name(model, name)
    
    module.weight.grad[mask] = 0.
    if getattr(module, "bias", None) is not None:
        module.bias.grad[mask] = 0.


@torch.no_grad()
def rollback_module(model, name, mask, pre_optim_state):
    module = find_module_by_name(model, name)
    
    module.weight[mask] = pre_optim_state[f"{name}.weight"][mask]
    if getattr(module, "bias", None) is not None:
        module.bias[mask] = pre_optim_state[f"{name}.bias"][mask]


@torch.no_grad()
def find_module_by_name(model, name):
    module = model
    splitted_name = name.split(".")
    for idx, sub in enumerate(splitted_name):
        if idx < len(splitted_name):
            module = getattr(module, sub)
    
    return module
