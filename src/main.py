import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from filelock import FileLock
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from arg_parser import get_parser, int2bool
from data import get_data
from utils import set_seed, run, get_model


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12356')
    
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, config):
    set_seed(config.seed)
    
    # Setup for multi gpu
    if rank > -1:
        print(f'=> Running training on rank {rank}')
        setup(rank, config.world_size)
        device = rank
    else:
        device = config.device
    
    # Get model
    model, hooks, arch_config = get_model(config)
    model.to(device)
    
    if rank > -1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Define optimizer and scheduler
    optimizer = SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
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
    
    previous_activations = {}
    grad_mask = {}
    
    # Train and test
    for epoch in range(config.epochs):
        if epoch > 0:
            run(config, model, valid_loader, None, scaler, device, arch_config, hooks)
            
            for k in hooks:
                activation_delta = torch.abs(previous_activations[k] - hooks[k].get_mean_activation())
                grad_mask[k] = torch.topk(activation_delta, k=int(config.topk * activation_delta.shape[0]),
                                          largest=False, sorted=False)[1]
                hooks[k].reset()
            
        train = run(config, model, train_loader, optimizer, scaler, device, arch_config, grad_mask)
        for k in hooks:
            hooks[k].reset()
            
        valid = run(config, model, valid_loader, None, scaler, device, arch_config, grad_mask)
        for k in hooks:
            previous_activations[k] = hooks[k].get_mean_activation()
            hooks[k].reset()
            
        test = run(config, model, test_loader, None, scaler, device, arch_config, grad_mask)
        for k in hooks:
            hooks[k].reset()
        
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
    
    config.amp = int2bool(config.amp)
    config.mask_gradients = int2bool(config.mask_gradients)
    
    config.world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(config)
    
    if config.world_size == 1:
        main(-1, config)
    else:
        mp.spawn(main, args=(config,), nprocs=config.world_size, join=True)
