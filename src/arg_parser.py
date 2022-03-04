import argparse


def int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1


# noinspection PyTypeChecker
def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # General
    parser.add_argument("--seed", type=int, default=1,
                        help="Reproducibility seed.")
    parser.add_argument("--root", type=str, default="/data/classification",
                        help="Dataset root folder.")
    parser.add_argument("--amp", type=int2bool, choices=[0, 1], default=True,
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
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="cifar10",
                        help="Source dataset.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers (threads) per process.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size per process.")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation size as portion of the whole train set.")
    
    # Pruning
    parser.add_argument("--rollback", type=int2bool, choices=[0, 1], default=False,
                        help="Rollback the weights update (removes momentum and wd effects).")
    parser.add_argument("--topk", type=float, default=0.5,
                        help="Topk percentage of gradients to retain.")
    parser.add_argument("--random-mask", type=int2bool, choices=[0, 1], default=False,
                        help="Apply a random gradient mask.")
    parser.add_argument("--mask-mode", type=str, choices=["per-sample", "per-feature"], default="per-sample",
                        help="Mask evaluation mode.")
    parser.add_argument("--delta-mode", type=str, choices=["difference", "cosine"], default="difference",
                        help="How to evaluate activations deltas.")
    
    return parser
