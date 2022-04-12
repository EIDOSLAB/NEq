import datetime
import os
import time

import numpy as np
import torch
import torch.utils.data
import torchvision
from torch import nn
from torchvision.transforms import functional as F, InterpolationMode

import general_utils
import wandb
from general_utils import set_seed, attach_hooks, get_gradient_mask
from optim import MaskedSGD, MaskedAdam
from segmentation import presets
from segmentation import utils
from segmentation.coco_utils import get_coco


def get_dataset(dir_path, name, image_set, val_size, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)
    
    paths = {
        "voc":     (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco":    (dir_path, get_coco, 21),
    }
    p, ds_fn, num_classes = paths[name]
    
    ds = ds_fn(p, image_set=image_set, val_size=val_size, transforms=transform)
    return ds, num_classes


def get_transform(train, args):
    if train:
        return presets.SegmentationPresetTrain(base_size=520, crop_size=480)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        
        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)
        
        return preprocessing
    else:
        return presets.SegmentationPresetEval(base_size=520)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
    
    if len(losses) == 1:
        return losses["out"]
    
    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, data_loader, device, num_classes, amp):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            print(image.shape)
            
            with torch.cuda.amp.autocast(enabled=amp):
                output = model(image)
                output = output["out"]
            
            confmat.update(target.flatten(), output.argmax(1).flatten())
        
        confmat.reduce_from_all_processes()
    
    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def activate_hooks(hooks, active):
    for h in hooks:
        hooks[h].activate(active)


def main(config):
    set_seed(config.seed)
    
    utils.init_distributed_mode(config)
    print(config)
    
    device = torch.device(config.device)
    
    dataset, num_classes = get_dataset(config.data_path, config.dataset, "train", config.val_size,
                                       (get_transform(True, config), get_transform(False, config)))
    dataset_test, _ = get_dataset(config.data_path, config.dataset, "val", config.val_size,
                                  get_transform(False, config))
    
    print(f"Train set length {len(dataset[0])}")
    print(f"Validation set length {len(dataset[1])}")
    print(f"Test set length {len(dataset_test)}")
    
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset[0])
        valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset[1])
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset[0])
        valid_sampler = torch.utils.data.RandomSampler(dataset[1])
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader = torch.utils.data.DataLoader(
        dataset[0],
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )
    
    data_loader_valid = torch.utils.data.DataLoader(
        dataset[1], batch_size=1, sampler=valid_sampler, num_workers=config.workers, collate_fn=utils.collate_fn
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=config.workers, collate_fn=utils.collate_fn
    )
    
    model = torchvision.models.segmentation.__dict__[config.arch](num_classes=num_classes, aux_loss=config.aux_loss)
    
    total_neurons = 0
    
    for m in model.modules():
        if hasattr(m, "weight"):
            total_neurons += m.weight.shape[0]
    
    model.to(device)
    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module
    
    params_to_optimize = [
        {"params": [p for n, p in model_without_ddp.backbone.named_parameters() if p.requires_grad],
         "names":  [n for n, p in model_without_ddp.backbone.named_parameters() if p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.classifier.named_parameters() if p.requires_grad],
         "names":  [n for n, p in model_without_ddp.classifier.named_parameters() if p.requires_grad]},
    ]
    if config.aux_loss:
        params = [p for n, p in model_without_ddp.aux_classifier.named_parameters() if p.requires_grad]
        names = [n for n, p in model_without_ddp.aux_classifier.named_parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "names": names, "lr": config.lr * 10})
    optimizer = MaskedSGD(params_to_optimize, [], lr=config.lr, momentum=config.momentum,
                          weight_decay=config.weight_decay)
    
    scaler = torch.cuda.amp.GradScaler() if config.amp else None
    
    iters_per_epoch = len(data_loader)
    main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (iters_per_epoch * (config.epochs - config.lr_warmup_epochs))) ** 0.9
    )
    
    if config.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * config.lr_warmup_epochs
        config.lr_warmup_method = config.lr_warmup_method.lower()
        if config.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=config.lr_warmup_decay, total_iters=warmup_iters
            )
        elif config.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=config.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{config.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler
    
    if config.resume:
        checkpoint = torch.load(config.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not config.test_only)
        if not config.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            config.start_epoch = checkpoint["epoch"] + 1
            if config.amp:
                scaler.load_state_dict(checkpoint["scaler"])
    
    if config.test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, amp=config.amp)
        print(confmat)
        return
    
    print("Initialize wandb run")
    wandb.init(project="zero-grad", config=config)
    os.makedirs(os.path.join("/scratch", "checkpoints", wandb.run.name), exist_ok=True)
    
    # Init dictionaries
    hooks = {}
    previous_activations = {}
    grad_mask = {}
    
    frozen_neurons = {"total": 0,
                      "layer": {f"{n}": 0 for n, m in model.named_modules() if
                                isinstance(m, (nn.Conv2d, nn.BatchNorm2d))}}
    
    # Attach the hooks used to gather the PSP value
    attach_hooks(config, model, hooks)
    
    evaluate(model, data_loader_valid, device=device, num_classes=num_classes, amp=config.amp)
    
    for k in hooks:
        previous_activations[k] = hooks[k].get_samples_activation()
        hooks[k].reset(previous_activations[k])
    
    start_time = time.time()
    for epoch in range(config.start_epoch, config.epochs):
        if epoch > (config.warmup - 1):
            # Log the amount of frozen neurons
            frozen_neurons = general_utils.log_masks(model, grad_mask, total_neurons)
        
        if config.distributed:
            train_sampler.set_epoch(epoch)
        
        activate_hooks(hooks, False)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, config.print_freq,
                        scaler)
        
        activate_hooks(hooks, True)
        confmat_val = evaluate(model, data_loader_valid, device=device, num_classes=num_classes, amp=config.amp)
        
        if config.rollback == "optim" and isinstance(optimizer, (MaskedSGD, MaskedAdam)):
            optimizer.param_groups[0]["masks"] = grad_mask
        
        # Save the activations into the dict
        log_deltas = {}
        for k in hooks:
            # Get the masks, either random or evaluated
            deltas = hooks[k].get_delta_of_delta() if config.delta_of_delta else hooks[k].get_reduced_activation_delta()
            get_gradient_mask(config, epoch + 1, k, deltas, grad_mask)
            hooks[k].reset()
            
            hist = np.histogram(deltas.cpu().numpy(), bins=min(512, deltas.shape[0]))
            log_deltas[f"{k}"] = wandb.Histogram(np_histogram=hist)
        
        activate_hooks(hooks, False)
        confmat_test = evaluate(model, data_loader_test, device=device, num_classes=num_classes, amp=config.amp)
        print(confmat_val)
        print(confmat_test)
        
        acc_global_val, acc_val, iu_val = confmat_val.compute()
        valid = {"acc_global": acc_global_val, "acc": acc_val, "iu": iu_val}
        acc_global_test, acc_test, iu_test = confmat_test.compute()
        test = {"acc_global": acc_global_test, "acc": acc_test, "iu": iu_test}
        
        wandb.log({
            "frozen_neurons_perc": frozen_neurons,
            # "train":               train,
            "valid":               valid,
            "test":                test,
            "epochs":              epoch,
            "lr":                  optimizer.param_groups[0]["lr"],
            "deltas":              log_deltas
        })
        
        checkpoint = {
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch":        epoch,
            "config":       config,
        }
        if config.amp:
            checkpoint["scaler"] = scaler.state_dict()
        
        torch.save(checkpoint, os.path.join("/scratch", "checkpoints", wandb.run.name, "checkpoint.pt"))
        del checkpoint
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    def int2bool(i):
        i = int(i)
        assert i == 0 or i == 1
        return i == 1
    
    def evaleps(i):
        if i == "none":
            return "none"
        else:
            return float(i)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)
    
    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--arch", default="fcn_resnet101", type=str, help="model name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
    
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")
    
    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    
    parser.add_argument("--seed", type=int, default=1,
                        help="Reproducibility seed.")
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
    parser.add_argument("--delta-of-delta", type=int2bool, choices=[0, 1], default=0,
                        help="Use delta of delta.")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.optim = "sgd"
    main(args)
