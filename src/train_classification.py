import os
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from filelock import FileLock
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from classification.arg_parser import get_parser
from classification.data import get_data
from classification.fit import run
from classification.models import get_model
from classification.utils import get_optimizer, get_scheduler
from general_utils import set_seed, setup, cleanup, attach_hooks, get_gradient_mask, log_masks, cosine_similarity
from optim import MaskedSGD, MaskedAdam


def activate_hooks(hooks, active):
    for h in hooks:
        hooks[h].activate(active)


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
    
    # Build optimizer and scheduler
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    if config.ckp:
        print(f"Loading from {config.ckp}")
        ckp = torch.load(config.ckp, map_location=device)
        model.load_state_dict(ckp["model"])
        optimizer.load_state_dict(ckp["optim"])
        
        ckp_epoch = ckp["epoch"]
        
        del ckp
    
    # DDP for multi gpu
    if rank > -1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Create dataloaders
    with FileLock('data.lock'):
        train_loader, valid_loader, test_loader = get_data(config)
    
    # Initialize amp
    scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda" and config.amp))
    
    # Init wandb
    if rank <= 0:
        print("Initialize wandb run")
        wandb.init(project=config.project_name, config=config)
        os.makedirs(os.path.join("/scratch", "checkpoints", wandb.run.id))
    
    # Init dictionaries
    hooks = {}
    previous_activations = {}
    grad_mask = {}
    
    frozen_neurons = {"total": 0,
                      "layer": {f"{n}": 0 for n, m in model.named_modules() if
                                isinstance(m, (nn.Conv2d, nn.BatchNorm2d))}}
    
    # Attach the hooks used to gather the PSP value
    if not config.param_norm:
        attach_hooks(config, model, hooks)
    
    # First run on validation to get the PSP for epoch -1
    activate_hooks(hooks, True)
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
    
    if config.param_norm:
        previous_params = {}
        for n, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                previous_params[n] = m.weight.view(m.weight.shape[0], -1).detach().clone()
    
    # In case of DDP wait for all the processes before starting the training
    if rank > -1:
        dist.barrier()
    
    train, valid, test = {}, {}, {}
    
    # Epochs cycle
    for epoch in range(config.epochs):
        
        if config.ckp and epoch < ckp_epoch:
            scheduler.step()
            continue
        
        if epoch > (config.warmup - 1):
            # Log the amount of frozen neurons
            frozen_neurons = log_masks(model, grad_mask, total_neurons)
        
        # Train step
        activate_hooks(hooks, False)
        train = run(config, model, train_loader, optimizer, scaler, device, grad_mask)
        
        # Gather the PSP values for the current epoch (after the train step)
        # attach_hooks(config, model, hooks)
        
        activate_hooks(hooks, True)
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
            for group in optimizer.param_groups:
                group["masks"] = grad_mask
        
        # Save the activations into the dict
        log_deltas = {"phi": {}, "d_phi": {}, "velocity": {}}
        for k in hooks:
            # Get the masks, either random or evaluated
            if config.delta_of_delta:
                deltas = hooks[k].get_delta_of_delta()
            elif config.velocity:
                deltas = hooks[k].get_velocity()
            else:
                deltas = hooks[k].get_reduced_activation_delta()
            
            phi = hooks[k].get_reduced_activation_delta()
            d_phi = hooks[k].get_delta_of_delta()
            velocity = hooks[k].get_velocity()
            
            get_gradient_mask(config, epoch + 1, k, deltas, grad_mask)
            
            if config.delta_of_delta or config.velocity:
                hooks[k].update_velocity()
                hooks[k].update_delta_buffer()
            
            hooks[k].reset()
            
            log_deltas["phi"][f"{k}"] = wandb.Histogram(np_histogram=np.histogram(phi.cpu().numpy(),
                                                                                  bins=min(512, phi.shape[0])))
            log_deltas["d_phi"][f"{k}"] = wandb.Histogram(np_histogram=np.histogram(d_phi.cpu().numpy(),
                                                                                    bins=min(512, d_phi.shape[0])))
            log_deltas["velocity"][f"{k}"] = wandb.Histogram(np_histogram=np.histogram(velocity.cpu().numpy(),
                                                                                       bins=min(512,
                                                                                                velocity.shape[0])))
        
        if config.param_norm:
            log_param_norm = {}
            for n, m in model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                    norm = 1 - cosine_similarity(previous_params[n],
                                                 (m.weight.detach().clone().view(m.weight.shape[0], -1)),
                                                 dim=1)
                    mask = torch.where(torch.abs(norm) <= config.eps)[0]
                    grad_mask[n] = mask
                    log_param_norm[n] = wandb.Histogram(np_histogram=np.histogram(norm.cpu().numpy(),
                                                                                  bins=min(512, norm.shape[0])))
                    
                    previous_params[n] = m.weight.detach().clone().view(m.weight.shape[0], -1)
        
        # Test step
        activate_hooks(hooks, False)
        test = run(config, model, test_loader, None, scaler, device, grad_mask)
        
        # Logs
        if rank <= 0:
            wandb.log({
                "frozen_neurons_perc": frozen_neurons,
                "train":               train,
                "valid":               valid,
                "test":                test,
                "epochs":              epoch,
                "lr":                  optimizer.param_groups[0]["lr"],
                "deltas":              log_deltas,
                "param_norm":          log_param_norm
            })
            
            print(f"Epoch\t {epoch}\n"
                  f"train\t {train}\n"
                  f"valid\t {valid}\n"
                  f"test\t {test}\n")
        
        checkpoint = {
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch":        epoch,
            "config":       config,
        }
        if config.amp:
            checkpoint["scaler"] = scaler.state_dict()
        
        torch.save(checkpoint, os.path.join("/scratch", "checkpoints", wandb.run.id, "checkpoint.pt"))
        del checkpoint
        
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
