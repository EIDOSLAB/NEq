import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from filelock import FileLock
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import models
import wandb
from arg_parser import get_parser
from data import get_data
from fit import run
from models import get_model, attach_hooks
from optim import MaskedSGD, MaskedAdam
from utils import set_seed, setup, get_optimizer, cleanup, get_gradient_mask, log_masks, get_scheduler


def main(rank, config):
    # Set reproducibility
    set_seed(config.seed)
    
    # Setup for multi gpu
    if rank > -1:
        print(f'=> Running training on rank {rank}')
        setup(rank, config.world_size)
        device = rank
    else:
        device = config.device
    
    # Get model
    model, total_neurons = get_model(config)
    model.to(device)
    
    # DDP for multi gpu
    if rank > -1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Build optimizer and scheduler
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # Create dataloaders
    with FileLock('data.lock'):
        train_loader, valid_loader, test_loader = get_data(config)
    
    # Initialize amp
    scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda" and config.amp))
    
    # Init wandb
    if rank <= 0:
        print("Initialize wandb run")
        wandb.init(project="zero-grad-test", config=config)
    
    # Init dictionaries
    hooks = {}
    previous_activations = {}
    grad_mask = {}
    
    frozen_neurons = {"total": 0,
                      "layer": {f"{n}": 0 for n, m in model.named_modules() if
                                isinstance(m, (nn.Conv2d, nn.BatchNorm2d))}}
    
    # Attach the hooks used to gather the PSP value
    attach_hooks(config, model, hooks)
    
    # First run on validation to get the PSP for epoch -1
    models.active = True
    valid = run(config, model, valid_loader, None, scaler, device, grad_mask)
    
    # If we want to rollback the model config we save the first configuration
    if config.rollback_model:
        best_epoch = -1
        best_valid_loss = valid["loss"]
        best_model_state_dict = deepcopy(model.state_dict())
        best_optim_state_dict = deepcopy(optimizer.state_dict())
    
    # Save the activations into the dict
    for k in hooks:
        previous_activations[k] = hooks[k].get_samples_activation()
        hooks[k].reset(previous_activations[k])
    
    # In case of DDP wait for all the processes before starting the training
    if rank > -1:
        dist.barrier()
    
    train, valid, test = {}, {}, {}
    
    # Epochs cycle
    for epoch in range(config.epochs):
        
        if epoch > (config.warmup - 1):
            # Log the amount of frozen neurons
            frozen_neurons = log_masks(model, grad_mask, total_neurons)
        
        # Train step
        # models.active = False
        # train = run(config, model, train_loader, optimizer, scaler, device, grad_mask)
        
        # Gather the PSP values for the current epoch (after the train step)
        # attach_hooks(config, model, hooks)

        models.active = True
        valid = run(config, model, valid_loader, None, scaler, device, grad_mask)
        
        # If we want to rollback the model config we update the configuration if the loss improved
        if config.rollback_model and valid["loss"] < best_valid_loss:
            best_epoch = epoch
            best_valid_loss = valid["loss"]
            best_model_state_dict = deepcopy(model.state_dict())
            best_optim_state_dict = deepcopy(optimizer.state_dict())
        
        # If we do not want to pin the frozen neurons, we reinitialize the masks dict
        if not config.pinning:
            grad_mask = {}
        
        # If we use the MaskedSGD optimizer we replace the mask used in the last epoch with an empty one.
        # It will be filled later
        if config.rollback == "optim" and isinstance(optimizer, (MaskedSGD, MaskedAdam)):
            optimizer.param_groups[0]["masks"] = grad_mask
        
        # Save the activations into the dict
        for k in hooks:
            # Get the masks, either random or evaluated
            deltas = hooks[k].get_delta_of_delta() if config.delta_of_delta else hooks[k].get_reduced_activation_delta()
            get_gradient_mask(config, epoch + 1, k, deltas, grad_mask)
            hooks[k].reset()
        
        # Test step
        models.active = False
        test = run(config, model, test_loader, None, scaler, device, grad_mask)
        
        # Logs
        if rank <= 0:
            wandb.log({
                "frozen_neurons_perc": frozen_neurons,
                "train":               train,
                "valid":               valid,
                "test":                test,
                "epochs":              epoch,
                "lr":                  optimizer.param_groups[0]["lr"]
            })
            
            print(f"Epoch\t {epoch}\n"
                  f"train\t {train}\n"
                  f"valid\t {valid}\n"
                  f"test\t {test}\n")
        
        # Scheduler step
        if scheduler is not None:
            # Rollback the model configuration
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
    # Get config
    config = get_parser()
    
    # Set WORLD_SIZE (given as env var) for multi-gpu training
    config.world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(config)
    
    # Call main()
    if config.world_size == 1:
        main(-1, config)
    else:
        mp.spawn(main, args=(config,), nprocs=config.world_size, join=True)
