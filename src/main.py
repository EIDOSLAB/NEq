import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from filelock import FileLock
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR

import wandb
from arg_parser import get_parser
from data import get_data
from fit import run
from models import get_model, attach_hooks
from utils import set_seed, setup, get_optimizer, cleanup, get_mask


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
    
    # Build optimizer and scheduler
    optimizer = get_optimizer(config, model)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150])
    
    # Create dataloaders
    with FileLock('data.lock'):
        train_loader, valid_loader, test_loader = get_data(config)
    
    # Initialize amp
    scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda" and config.amp))
    
    # Init wandb
    if rank <= 0:
        wandb.init(project="zero-grad", config=config)
    
    pre_epoch_activations = {}
    post_epoch_activations = {}
    hooks = {}
    
    # Get the activations for "epoch" -1
    attach_hooks(config, model, hooks)
    
    valid = run(config, model, valid_loader, None, scaler, device, arch_config)
    
    if config.rollback_model:
        best_epoch = -1
        best_valid_loss = valid["loss"]
        best_model_state_dict = deepcopy(model.state_dict())
        best_optim_state_dict = deepcopy(optimizer.state_dict())
    
    for k in hooks:
        pre_epoch_activations[k] = hooks[k].get_samples_activation()
        hooks[k].close()
    
    train, valid, test = {}, {}, {}
    grad_mask = {}
    
    if rank > -1:
        dist.barrier()
    
    # Train and test
    for epoch in range(config.epochs):
        # Do this only if we freeze some neurons
        if not config.pinning:
            grad_mask = {}
        # If we use the MaskedSGD optimizer we replace the mask used in the last epoch with an empty one.
        # It will be filled later
        if config.rollback == "optim":
            optimizer.param_groups[0]["masks"] = grad_mask
        
        # Get the neurons masks
        if len(post_epoch_activations):
            total_neurons = 0
            frozen_neurons = 0
            for k in hooks:
                # Get the masks, either random or evaluated
                get_mask(config, epoch, k, pre_epoch_activations, post_epoch_activations, grad_mask, arch_config)
                
                total_neurons += pre_epoch_activations[k].shape[1]
                frozen_neurons += grad_mask[k].shape[0]
                
                wandb.log({f"frozen_neurons_perc_{k}": grad_mask[k].shape[0] / pre_epoch_activations[k].shape[1] * 100})
                
                # Update the activations dictionary
                pre_epoch_activations[k] = post_epoch_activations[k]
            
            wandb.log({f"frozen_neurons_perc": frozen_neurons / total_neurons * 100})
        
        # Train step
        train = run(config, model, train_loader, optimizer, scaler, device, grad_mask)
        
        # Gather the activations values for the current epoch (after the train step)
        attach_hooks(config, model, hooks)
        
        valid = run(config, model, valid_loader, None, scaler, device, grad_mask)
        
        if config.rollback_model and valid["loss"] < best_valid_loss:
            best_epoch = epoch
            best_valid_loss = valid["loss"]
            best_model_state_dict = deepcopy(model.state_dict())
            best_optim_state_dict = deepcopy(optimizer.state_dict())
        
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
        
            print(f"Epoch\t {epoch}\n"
                  f"train\t {train}\n"
                  f"valid\t {valid}\n"
                  f"test\t {test}\n")
        
        if scheduler is not None:
            if config.rollback_model and (((epoch + 1) == 100) or ((epoch + 1) == 150)):
                print(f"Rollback to best_model_state_dict (epoch {best_epoch})")
                model.load_state_dict(best_model_state_dict)
                optimizer.load_state_dict(best_optim_state_dict)
            scheduler.step()
        
        if rank > -1:
            dist.barrier()
    
    if rank <= 0:
        wandb.run.finish()
    
    cleanup()
    exit(0)


if __name__ == '__main__':
    config = get_parser()
    
    config.world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(config)
    
    if config.world_size == 1:
        main(-1, config)
    else:
        mp.spawn(main, args=(config,), nprocs=config.world_size, join=True)
