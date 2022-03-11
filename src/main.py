import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from filelock import FileLock
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR

import wandb
from arg_parser import get_parser, int2bool
from data import get_data
from optim import MaskedSGD
from utils import set_seed, run, get_model, Hook


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12356')
    
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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
    # zeros_1 = torch.nonzero(mask_1)
    mask_2 = (torch.abs(x1).sum(dim=dim) != 0) * (torch.abs(x2).sum(dim=dim) != 0)
    # zeros_2 = torch.nonzero(mask_2)
    
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


def evaluate_mask(config, epoch, k, pre_epoch_activations, post_epoch_activations, topk, grad_mask, arch_config):
    if config.delta_mode == "difference":
        activation_delta = torch.abs(pre_epoch_activations[k].float() - post_epoch_activations[k].float())
    if config.delta_mode == "cosine":
        shape = pre_epoch_activations[k].shape
        activation_delta = 1 - cosine_similarity(pre_epoch_activations[k].float().view(shape[0], shape[1], -1),
                                                 post_epoch_activations[k].float().view(shape[0], shape[1], -1),
                                                 dim=2)
    
    reduced_activation_delta = reduce_delta(config, activation_delta)
    
    hist = np.histogram(reduced_activation_delta.cpu().numpy(), bins=min(512, reduced_activation_delta.shape[0]))
    wandb.log({f"mean_deltas_{k}": wandb.Histogram(np_histogram=hist)})

    # If the warmup epochs are over we can start evaluating the masks
    if epoch > (config.warmup - 1):
        mask = torch.topk(reduced_activation_delta, k=topk, largest=False, sorted=False)[1]
        grad_mask[k] = mask
        for attached in arch_config["bn-conv"][k]:
            grad_mask[attached] = mask


def get_random_mask(config, epoch, k, pre_epoch_activations, topk, grad_mask, arch_config):

    # If the warmup epochs are over we can start evaluating the masks
    if epoch > (config.warmup - 1):
        mask = random.sample(range(0, pre_epoch_activations[k].shape[1]), topk)
        grad_mask[k] = mask
        for attached in arch_config["bn-conv"][k]:
            grad_mask[attached] = mask


def main(rank, config):
    # Set reproducibility env
    set_seed(config.seed)
    
    # Setup for multi gpu
    if rank > -1:
        print(f'=> Running training on rank {rank}')
        setup(rank, config.world_size)
        device = rank
    else:
        device = config.device
    
    # Get model
    model, arch_config = get_model(config)
    model.to(device)
    
    # DDP for multi gpu
    if rank > -1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Define optimizer and scheduler
    named_params = list(map(list, zip(*list(model.named_parameters()))))
    optimizer = MaskedSGD(named_params[1], names=named_params[0], lr=config.lr, weight_decay=config.weight_decay,
                          momentum=config.momentum)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150])
    
    # Create dataloaders
    with FileLock('data.lock'):
        train_loader, valid_loader, test_loader = get_data(config)
    
    # Initialize amp
    scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda" and config.amp))
    
    if rank > -1:
        dist.barrier()
    
    # Init wandb
    if rank <= 0:
        wandb.init(project="zero-grad", config=config)
    
    pre_epoch_activations = {}
    post_epoch_activations = {}
    hooks = {}
    
    # Get the activations for "epoch" -1
    for n, m in model.named_modules():
        if n in arch_config["targets"]:
            hooks[n] = Hook(config, n, m)
    
    run(config, model, valid_loader, None, scaler, device, arch_config)
    
    for k in hooks:
        pre_epoch_activations[k] = hooks[k].get_samples_activation()
        hooks[k].close()
    
    train, valid, test = {}, {}, {}
    
    # Train and test
    for epoch in range(config.epochs):
        # Do this only if we freeze some neurons
        if config.topk < 1:
            # If we use the MaskedSGD optimizer we set the replace the mask used in the last epoch with an empty one.
            # It will be filled later
            if config.rollback == "optim":
                grad_mask = {}
                optimizer.param_groups[0]["masks"] = grad_mask
            
            # Get the neurons masks
            if len(post_epoch_activations):
                for k in hooks:
                    # How many neurons to select as "to freeze" as percentage of the total number of neurons
                    topk = int((1 - config.topk) * pre_epoch_activations[k].shape[1])
                    
                    # Get the masks, either random or evaluated
                    if config.random_mask:
                        get_random_mask(config, epoch, k, pre_epoch_activations, topk, grad_mask, arch_config)
                    else:
                        evaluate_mask(config, epoch, k, pre_epoch_activations, post_epoch_activations,
                                      topk, grad_mask, arch_config)
                    
                    # Update the activations dictionary
                    pre_epoch_activations[k] = post_epoch_activations[k]
        
        # Train step
        train = run(config, model, train_loader, optimizer, scaler, device, grad_mask)
        
        # Gather the activations values for the current epoch (after the train step)
        for n, m in model.named_modules():
            if n in arch_config["targets"]:
                hooks[n] = Hook(config, n, m)
        
        valid = run(config, model, valid_loader, None, scaler, device, grad_mask)
        
        for k in hooks:
            post_epoch_activations[k] = hooks[k].get_samples_activation()
            hooks[k].close()
        
        # Test step
        test = run(config, model, test_loader, None, scaler, device, grad_mask)
        
        # Logs
        if rank <= 0:
            wandb.log({
                "train":  train,
                "valid":  valid,
                "test":   test,
                "epochs": epoch,
                "lr":     optimizer.param_groups[0]["lr"]
            })
        
        if scheduler is not None:
            scheduler.step()
        
        if rank > -1:
            dist.barrier()
    
    if rank <= 0:
        wandb.run.finish()
    
    cleanup()
    exit(0)


if __name__ == '__main__':
    config = get_parser().parse_args()
    
    if isinstance(config.amp, int):
        config.amp = int2bool(config.amp)
    if isinstance(config.random_mask, int):
        config.random_mask = int2bool(config.random_mask)
    
    config.world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(config)
    
    if config.world_size == 1:
        main(-1, config)
    else:
        mp.spawn(main, args=(config,), nprocs=config.world_size, join=True)
