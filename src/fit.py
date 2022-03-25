from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from utils import find_module_by_name


def topk_accuracy(outputs, labels, topk=1):
    outputs = torch.softmax(outputs, dim=1)
    _, preds = outputs.topk(topk, dim=1)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds)).sum()
    return (correct / float(len(outputs))).cpu().item()


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


def run(config, model, dataloader, optimizer, scaler, device, grad_mask):
    train = optimizer is not None
    
    tot_loss = 0.
    outputs = []
    targets = []
    
    model.train(train)
    pbar = tqdm(dataloader, desc="Training" if train else "Testing",
                disable=(dist.is_initialized() and dist.get_rank() > 0))
    
    if config.dataset == "cifar10":
        target_bs = 100
    elif config.dataset == "imagenet":
        target_bs = 256

    iters_to_accumulate = target_bs // config.batch_size
    
    if train:
        optimizer.zero_grad()
    
    for batch, (images, target) in enumerate(pbar):
        images, target = images.to(device, non_blocking=True), \
                         target.to(device, non_blocking=True)
        
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(config.device == "cuda" and config.amp)):
                output = model(images)
                
                loss = F.cross_entropy(output, target)
                
                if train:
                    scaler.scale(loss).backward()
                    
                    if config.rollback == "manual":
                        pre_optim_state = deepcopy(model.state_dict())
                    elif config.rollback == "none":
                        for k in grad_mask:
                            zero_gradients(model, k, grad_mask[k])
                            
                    if ((batch + 1) % iters_to_accumulate == 0) or ((batch + 1) == len(dataloader)):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    if config.rollback == "manual":
                        for k in grad_mask:
                            rollback_module(model, k, grad_mask[k], pre_optim_state)
        
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
