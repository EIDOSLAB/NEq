import os
import random

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
        self.output_sum = 0
        self.active_count = torch.zeros(module.weight.shape[0], device=config.device)
        
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        relu_output = F.relu(output).mean(dim=(2, 3))
        self.active_count += relu_output.bool().sum(dim=0)
        self.output_sum = relu_output.sum(dim=0)
    
    def get_mean_activation(self):
        return self.output_sum / self.active_count
    
    def reset(self):
        self.output_sum = 0
        self.active_count.mul_(0)
    
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
    
    hooks = {}
    
    for n, m in model.named_modules():
        if n in arch_config["targets"]:
            hooks[n] = Hook(config, n, m)
    
    return model, hooks, arch_config


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


def run(config, model, dataloader, optimizer, scaler, device, arch_config, grad_mask):
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
                    
                    if config.mask_gradients:
                        for k in grad_mask:
                            zero_gradients(model, k.split("."), grad_mask[k])
                            zero_gradients(model, arch_config["bn-conv"][k].split("."), grad_mask[k])
                    
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
    module = model
    for idx, sub in enumerate(name):
        if idx < len(name):
            module = getattr(module, sub)
    
    module.weight.grad[mask] = 0.
    if getattr(module, "bias", None) is not None:
        module.bias.grad[mask] = 0.
