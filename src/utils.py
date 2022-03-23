import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import Adam

import wandb
from optim import MaskedSGD


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
    # Define optimizer and scheduler
    if config.optim == "sgd":
        named_params = list(map(list, zip(*list(model.named_parameters()))))
        return MaskedSGD(named_params[1], names=named_params[0], lr=config.lr, weight_decay=config.weight_decay,
                         momentum=config.momentum)
    if config.optim == "adam":
        return Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def get_gradient_mask(config, epoch, k, pre_epoch_activations, post_epoch_activations, grad_mask, arch_config):
    # If the warmup epochs are over we can start evaluating the masks
    if epoch > (config.warmup - 1):
        if config.random_mask:
            random_mask(k, pre_epoch_activations, config.topk, grad_mask, arch_config)
        else:
            evaluated_mask(config, k, pre_epoch_activations, post_epoch_activations, config.topk, grad_mask,
                           arch_config)


def random_mask(k, pre_epoch_activations, topk, grad_mask, arch_config):
    # How many neurons to select as "to freeze" as percentage of the total number of neurons
    topk = int((1 - topk) * pre_epoch_activations[k].shape[1])
    
    mask = random.sample(range(0, pre_epoch_activations[k].shape[1]), topk)
    
    grad_mask[k] = mask
    # for attached in arch_config["bn-conv"][k]:
    #     grad_mask[attached] = mask


def evaluated_mask(config, k, pre_epoch_activations, post_epoch_activations, topk, grad_mask, arch_config):
    reduced_activation_delta = get_reduced_activation_delta(config, k, pre_epoch_activations, post_epoch_activations)
    
    if config.eps != "-":
        mask = torch.where(reduced_activation_delta <= config.eps)[0]
    elif config.binomial:
        mask = torch.where(torch.distributions.binomial.Binomial(probs=reduced_activation_delta).sample() == 0)[0]
    else:
        # How many neurons to select as "to freeze" as percentage of the total number of neurons
        topk = int((1 - topk) * pre_epoch_activations[k].shape[1])
        mask = torch.topk(reduced_activation_delta, k=topk, largest=False, sorted=False)[1]
    
    if config.pinning and k in grad_mask:
        grad_mask[k] = torch.cat([grad_mask[k].long(), mask.long()]).unique()
        # for attached in arch_config["bn-conv"][k]:
        #     grad_mask[attached] = torch.cat([grad_mask[attached].long(), mask.long()]).unique()
    else:
        grad_mask[k] = mask
        # for attached in arch_config["bn-conv"][k]:
        #     grad_mask[attached] = mask
    
    hist = np.histogram(reduced_activation_delta.cpu().numpy(), bins=min(512, reduced_activation_delta.shape[0]))
    wandb.log({f"mean_deltas_{k}": wandb.Histogram(np_histogram=hist)})


def get_reduced_activation_delta(config, k, pre_epoch_activations, post_epoch_activations):
    if config.delta_mode == "difference":
        activation_delta = torch.abs(pre_epoch_activations[k].float() - post_epoch_activations[k].float())
    if config.delta_mode == "cosine":
        shape = pre_epoch_activations[k].shape
        activation_delta = 1 - torch.abs(
            cosine_similarity(
                pre_epoch_activations[k].float().view(shape[0], shape[1], -1),
                post_epoch_activations[k].float().view(shape[0], shape[1], -1),
                dim=2
            )
        )
    
    return reduce_delta(config, activation_delta)


def cosine_similarity(x1, x2, dim, eps=1e-8):
    x1_squared_norm = torch.pow(x1, 2).sum(dim=dim, keepdim=True)
    x2_squared_norm = torch.pow(x2, 2).sum(dim=dim, keepdim=True)
    
    x1_squared_norm.clamp_min_(eps)
    x2_squared_norm.clamp_min_(eps)
    
    x1_norm = x1_squared_norm.sqrt_()
    x2_norm = x2_squared_norm.sqrt_()
    
    x1_normalized = x1.div(x1_norm)
    x2_normalized = x2.div(x2_norm)
    
    mask_1 = (torch.abs(x1).sum(dim=dim) == 0) * (torch.abs(x2).sum(dim=dim) == 0)
    mask_2 = (torch.abs(x1).sum(dim=dim) != 0) * (torch.abs(x2).sum(dim=dim) != 0)
    
    cos_sim_value = torch.sum(x1_normalized * x2_normalized, dim=dim)
    
    return mask_2 * cos_sim_value + mask_1


def reduce_delta(config, activation_delta):
    if config.mask_mode == "per-sample":
        if config.reduction == "mean":
            reduced_activation_delta = torch.mean(activation_delta, dim=0)
        elif config.reduction == "max":
            reduced_activation_delta = torch.max(activation_delta, dim=0)[0]
    elif config.mask_mode == "per-feature":
        if config.delta_mode == "difference":
            if config.reduction == "mean":
                reduced_activation_delta = torch.mean(activation_delta, dim=(0, 2, 3))
            elif config.reduction == "max":
                reduced_activation_delta = torch.max(activation_delta, dim=0)[0]
        elif config.delta_mode == "cosine":
            if config.reduction == "mean":
                reduced_activation_delta = torch.mean(activation_delta, dim=0)
            elif config.reduction == "max":
                reduced_activation_delta = torch.max(activation_delta, dim=0)[0]
    
    return reduced_activation_delta


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
    
    for k in grad_mask:
        frozen_neurons += grad_mask[k].shape[0]
        
        module = find_module_by_name(model, k)
        
        # Log the percentage of frozen neurons per layer
        wandb.log({f"frozen_neurons_perc_{k}": grad_mask[k].shape[0] / module.weight.shape[0] * 100})
    
    # Log the total percentage of frozen neurons
    wandb.log({f"frozen_neurons_perc": frozen_neurons / total_neurons * 100})
