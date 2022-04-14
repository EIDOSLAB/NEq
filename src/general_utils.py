import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch import Generator, nn
from torch.utils.data import random_split, Dataset


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


class MapDataset(Dataset):
    """Given a dataset, creates a dataset which applies a mapping function to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.

    Args:
        dataset:
        map_fn:
    """
    
    def __init__(self, dataset, map_fn, with_target=False):
        self.dataset = dataset
        self.map = map_fn
        self.with_target = with_target
    
    def __getitem__(self, index):
        if self.with_target:
            return self.map(self.dataset[index][0], self.dataset[index][1])
        else:
            return self.map(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


def split_dataset(dataset: torch.utils.data.Dataset, percentage: float, random_seed: int = 0) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Randomly splits a `torch.utils.data.Dataset` instance in two non-overlapping separated `Datasets`.

    The split of the elements of the original `Dataset` is based on `percentage` $$\in [0, 1]$$.
    I.e. if `percentage=0.2` the first returned dataset will contain 80% of the total elements and the second 20%.

    Args:
        dataset (torch.utils.data.Dataset): `torch.utils.data.Dataset` instance to be split.
        percentage (float): percentage of elements of `dataset` contained in the second dataset.
        random_seed (int): random seed for the split generator.

    Returns:
        tuple: a tuple containing the two new datasets.

    """
    dataset_length = len(dataset)
    valid_length = int(np.floor(percentage * dataset_length))
    train_length = dataset_length - valid_length
    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length],
                                                generator=Generator().manual_seed(random_seed))
    
    return train_dataset, valid_dataset


def attach_hooks(config, model, hooks):
    for n, m in model.named_modules():
        # if n in arch_config["targets"]:
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            hooks[n] = Hook(config, n, m)


class Hook:
    
    def __init__(self, config, name, module, momentum=0) -> None:
        self.name = name
        self.module = module
        self.samples_activation = []
        self.previous_activations = None
        self.activation_deltas = 0
        self.total_samples = 0
        
        self.momentum = momentum
        self.delta_buffer = 0
        
        self.config = config
        
        self.active = True
        
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        
        if not self.active:
            return
        
        # TODO sort this mess
        if self.config.mask_mode == "per-sample":
            reshaped_output = output.view(output.shape[0], output.shape[1], -1).mean(dim=2)
        if self.config.mask_mode == "per-feature":
            reshaped_output = output.view(output.shape[0], output.shape[1], -1)
        
        if self.config.dataset in ["imagenet", "coco"]:
            reshaped_output = reshaped_output.cpu()
        
        if self.previous_activations is None:
            self.samples_activation.append(reshaped_output)
        else:
            previous = self.previous_activations[self.total_samples:output.shape[0] + self.total_samples].float()
            if self.config.delta_mode == "difference":
                delta = reshaped_output.float() - previous
            if self.config.delta_mode == "cosine":
                delta = 1 - torch.abs(
                    cosine_similarity(
                        reshaped_output.float(),
                        previous,
                        dim=0 if self.config.mask_mode == "per-sample" else 2
                    )
                )
            
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
    
    def get_delta_of_delta(self):
        reduced_activation_delta = self.get_reduced_activation_delta()
        delta_of_delta = torch.abs(self.delta_buffer - reduced_activation_delta)
        self.delta_buffer = reduced_activation_delta
        
        return delta_of_delta
    
    def reset(self, previous_activations=None):
        self.samples_activation = []
        self.activation_deltas = 0
        self.total_samples = 0
        if previous_activations is not None:
            self.previous_activations = previous_activations
    
    def close(self) -> None:
        self.hook.remove()
        
    def activate(self, active):
        self.active = active


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
