import argparse
import sys


def int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1


def evaleps(i):
    if i == "none":
        return "none"
    else:
        return float(i)


# noinspection PyTypeChecker
def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # General
    parser.add_argument("--seed", type=int, default=1,
                        help="Reproducibility seed.")
    parser.add_argument("--root", type=str, default="/data/classification",
                        help="Dataset root folder.")
    parser.add_argument("--amp", type=int2bool, choices=[0, 1], default=1,
                        help="If True use torch.cuda.amp.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda",
                        help="Device type.")
    
    # Model
    parser.add_argument("--arch", type=str, default="resnet32-cifar",
                        help="Architecture name.")
    
    # Train
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of benchmark epochs.")
    
    # Optimizer
    parser.add_argument("--optim", type=str, choices=["sgd", "adam"], default="sgd",
                        help="Optimizer.")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Optimizer learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Optimizer momentum.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Optimizer weight decay.")
    
    # Dataset
    parser.add_argument("--dataset", type=str, choices=["cifar10", "imagenet"], default="cifar10",
                        help="Source dataset.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers (threads) per process.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size per process.")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation size as portion of the whole train set.")
    
    # Pruning
    parser.add_argument("--rollback", type=str, choices=["none", "manual", "optim"], default="optim",
                        help="Rollback the weights update (removes momentum and wd effects)."
                             "If `optim` also avoids momentum memorization for zeroed gradients")
    parser.add_argument("--topk", type=float, default=0,
                        help="Topk percentage of gradients to retain. Ignored if eps is not `none`. Set to 1 for baseline")
    parser.add_argument("--random-mask", type=int2bool, choices=[0, 1], default=0,
                        help="Apply a random gradient mask.")
    parser.add_argument("--mask-mode", type=str, choices=["per-sample", "per-feature"], default="per-feature",
                        help="Mask evaluation mode.")
    parser.add_argument("--delta-mode", type=str, choices=["difference", "cosine"], default="cosine",
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
    parser.add_argument("--velocity", type=int2bool, choices=[0, 1], default=1,
                        help="Use velocity.")
    parser.add_argument("--velocity-mu", type=float, default=0,
                        help="Velocity momentum")
    parser.add_argument("--param-norm", type=int2bool, default=0,
                        help="Use the norm of the parameters instead of the PSP")
    
    parser.add_argument("--ckp", type=str)
    parser.add_argument("--project-name", type=str, default="NEq")
    
    config = parser.parse_args()
    
    if config.delta_of_delta and config.velocity:
        print("Only one between delta-of-delta and velocity can be true")
        sys.exit(1)
    
    # Just for peace of mind in the wandb table
    if config.eps == "none":
        config.eps = "-"
    else:
        config.topk = "-"
    
    if config.binomial:
        config.eps = "-"
        config.topk = "-"
    
    return config
