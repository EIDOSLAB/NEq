import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR, StepLR

import wandb
from optim import MaskedSGD, MaskedAdam


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12356')
    
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_optimizer(config, model):
    print(f"Initialize optimizer {config.optim}")
    
    # Define optimizer and scheduler
    named_params = list(map(list, zip(*list(model.named_parameters()))))
    if config.optim == "sgd":
        return MaskedSGD(named_params[1], names=named_params[0], lr=config.lr, weight_decay=config.weight_decay,
                         momentum=config.momentum)
    if config.optim == "adam":
        return MaskedAdam(named_params[1], names=named_params[0], lr=config.lr, weight_decay=config.weight_decay)


def get_scheduler(config, optimizer):
    print("Initialize scheduler")
    
    if config.dataset == "cifar10":
        return MultiStepLR(optimizer, milestones=[100, 150])
    if config.dataset == "imagenet":
        return StepLR(optimizer, step_size=30)


def get_gradient_mask(config, epoch, k, reduced_activation_delta, grad_mask):
    # If the warmup epochs are over we can start evaluating the masks
    if epoch > (config.warmup - 1):
        if config.random_mask:
            random_mask(k, reduced_activation_delta, config.topk, grad_mask)
        else:
            evaluated_mask(config, k, reduced_activation_delta, config.topk, grad_mask)


def random_mask(k, reduced_activation_delta, topk, grad_mask):
    # How many neurons to select as "to freeze" as percentage of the total number of neurons
    topk = int((1 - topk) * reduced_activation_delta.shape[0])
    
    mask = random.sample(range(0, reduced_activation_delta.shape[0]), topk)
    
    grad_mask[k] = mask


def evaluated_mask(config, k, reduced_activation_delta, topk, grad_mask):
    if config.eps != "-":
        mask = torch.where(reduced_activation_delta <= config.eps)[0]
    elif config.binomial:
        mask = torch.where(torch.distributions.binomial.Binomial(probs=reduced_activation_delta).sample() == 0)[0]
    else:
        # How many neurons to select as "to freeze" as percentage of the total number of neurons
        topk = int((1 - topk) * reduced_activation_delta[k].shape[0])
        mask = torch.topk(reduced_activation_delta, k=topk, largest=False, sorted=False)[1]
    
    if config.pinning and k in grad_mask:
        grad_mask[k] = torch.cat([grad_mask[k].long(), mask.long()]).unique()
    else:
        grad_mask[k] = mask
    
    hist = np.histogram(reduced_activation_delta.cpu().numpy(), bins=min(512, reduced_activation_delta.shape[0]))
    wandb.log({f"mean_deltas_{k}": wandb.Histogram(np_histogram=hist)})


@torch.no_grad()
def find_module_by_name(model, name):
    module = model
    splitted_name = name.split(".")
    for idx, sub in enumerate(splitted_name):
        if idx < len(splitted_name):
            module = getattr(module, sub)
    
    return module


def log_masks(model, grad_mask, total_neurons):
    frozen_neurons = 0
    
    per_layer_frozen_neurons = {}
    
    for k in grad_mask:
        frozen_neurons += grad_mask[k].shape[0]
        
        module = find_module_by_name(model, k)
        
        # Log the percentage of frozen neurons per layer
        per_layer_frozen_neurons[f"{k}"] = grad_mask[k].shape[0] / module.weight.shape[0] * 100
    
    # Log the total percentage of frozen neurons
    return {"total": frozen_neurons / total_neurons * 100,
            "layer": per_layer_frozen_neurons}
