# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from torch import nn

from config import get_config
from general_utils import attach_hooks, get_gradient_mask, log_masks
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optim import MaskedSGD, MaskedAdam, MaskedAdamW
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def activate_hooks(hooks, active):
    for h in hooks:
        hooks[h].activate(active)


def int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1


def evaleps(i):
    if i == "none":
        return "none"
    else:
        return float(i)


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    
    # Added
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation size as portion of the whole train set.")
    
    # Pruning
    parser.add_argument("--rollback", type=str, choices=["none", "manual", "optim"], default="none",
                        help="Rollback the weights update (removes momentum and wd effects)."
                             "If `optim` also avoids momentum memorization for zeroed gradients")
    parser.add_argument("--topk", type=float, default=0.5,
                        help="Topk percentage of gradients to retain. Ignored if eps is not `none`. Set to 1 for baseline")
    parser.add_argument("--random-mask", type=int2bool, choices=[0, 1], default=0,
                        help="Apply a random gradient mask.")
    parser.add_argument("--mask-mode", type=str, choices=["per-sample", "per-feature"], default="per-sample",
                        help="Mask evaluation mode.")
    parser.add_argument("--delta-mode", type=str, choices=["difference", "cosine"], default="difference",
                        help="How to evaluate activations deltas.")
    parser.add_argument("--reduction", type=str, choices=["mean", "max"], default="mean",
                        help="Delta reduction on the sample dimension.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="How many warmup epochs.")
    parser.add_argument("--eps", type=evaleps, default="none",
                        help="Epsilon. `none` or float. If `none`, topk will be used to define the grad mask")
    parser.add_argument("--binomial", type=int2bool, choices=[0, 1], default=0,
                        help="Binomial for mask.")
    parser.add_argument("--pinning", type=int2bool, choices=[0, 1], default=0,
                        help="Pinning.")
    parser.add_argument("--rollback-model", type=int2bool, choices=[0, 1], default=0,
                        help="Rollback the model configuration before a decay step.")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--delta-of-delta", type=int2bool, choices=[0, 1], default=0,
                       help="Use delta of delta.")
    group.add_argument("--velocity", type=int2bool, choices=[0, 1], default=0,
                       help="Use velocity.")
    parser.add_argument("--velocity-mu", type=float, default=0,
                        help="Velocity momentum")
    
    parser.add_argument("--dataset", default="imagenet", type=str, help="dataset name")
    parser.add_argument("--arch", default="swin", type=str, help="model name")
    
    args, unparsed = parser.parse_known_args()
    
    config = get_config(args)
    
    return args, config


def main(args, config):
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_loader(
        config)
    
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))
    
    optimizer = build_optimizer(config, model)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    max_accuracy = 0.0
    
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_test, model)
        logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return
    
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model, logger)
        acc1, acc5, loss = validate(config, data_loader_test, model)
        logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {acc1:.1f}%")
    
    if config.THROUGHPUT_MODE:
        throughput(data_loader_test, model, logger)
        return
    
    print("Initialize wandb run")
    wandb.init(project="zero-grad", config=args)
    os.makedirs(os.path.join("/scratch", "checkpoints", wandb.run.name), exist_ok=True)
    
    # Init dictionaries
    hooks = {}
    previous_activations = {}
    grad_mask = {}
    
    frozen_neurons = {"total": 0,
                      "layer": {f"{n}": 0 for n, m in model.named_modules() if
                                isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.LayerNorm))}}
    
    total_neurons = 0
    
    for m in model.modules():
        if hasattr(m, "weight"):
            total_neurons += m.weight.shape[0]
    
    # Attach the hooks used to gather the PSP value
    attach_hooks(args, model, hooks, "head")
    
    validate(config, data_loader_val, model)
    
    for k in hooks:
        previous_activations[k] = hooks[k].get_samples_activation()
        hooks[k].reset(previous_activations[k])
    
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if epoch > (args.warmup - 1):
            # Log the amount of frozen neurons
            frozen_neurons = log_masks(model, grad_mask, total_neurons)
        data_loader_train.sampler.set_epoch(epoch)
        
        activate_hooks(hooks, False)
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger)
        
        activate_hooks(hooks, True)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        valid = {"accuracy": {"top1": acc1, "top5": acc5}, "loss": loss}
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        
        if args.rollback == "optim" and isinstance(optimizer, (MaskedSGD, MaskedAdam, MaskedAdamW)):
            for group in optimizer.param_groups:
                group["masks"] = grad_mask
        
        # Save the activations into the dict
        log_deltas = {}
        for k in hooks:
            # Get the masks, either random or evaluated
            if args.delta_of_delta:
                deltas = hooks[k].get_delta_of_delta()
            elif args.velocity:
                deltas = hooks[k].get_velocity()
            else:
                deltas = hooks[k].get_reduced_activation_delta()
            
            get_gradient_mask(args, epoch + 1, k, deltas, grad_mask)
            
            hooks[k].update_velocity()
            hooks[k].update_delta_buffer()
            hooks[k].reset()
            
            hist = np.histogram(deltas.cpu().numpy(), bins=min(512, deltas.shape[0]))
            log_deltas[f"{k}"] = wandb.Histogram(np_histogram=hist)
        
        activate_hooks(hooks, False)
        acc1, acc5, loss = validate(config, data_loader_test, model)
        test = {"accuracy": {"top1": acc1, "top5": acc5}, "loss": loss}
        logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {acc1:.1f}%")
        
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        
        wandb.log({
            "frozen_neurons_perc": frozen_neurons,
            # "train":               train,
            "valid":               valid,
            "test":                test,
            "epochs":              epoch,
            "lr":                  optimizer.param_groups[0]["lr"],
            "deltas":              log_deltas
        })
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        outputs = model(samples)
        
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)
        
        torch.cuda.synchronize()
        
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # compute output
        output = model(images)
        
        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()
    
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()
    
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.val_size = args.val_size
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()
    
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
    
    # print config
    logger.info(config.dump())

    args.optim = "adamw"

    # Just for peace of mind in the wandb table
    if args.eps == "none":
        args.eps = "-"
    else:
        args.topk = "-"

    if args.binomial:
        args.eps = "-"
        args.topk = "-"
    
    main(args, config)
