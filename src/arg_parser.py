import argparse


def int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # General
    parser.add_argument("--seed", type=int, default=1,
                        help="Reproducibility seed.")
    parser.add_argument("--root", type=str, default="/data/classification",
                        help="Dataset root folder.")
    parser.add_argument("--amp", type=int2bool, default=1,
                        help="If True use torch.cuda.amp.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda",
                        help="Device type.")
    
    # Model
    parser.add_argument("--arch", type=str, default="lenet5",
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
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="Optimizer weight decay.")
    
    # Dataset
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist",
                        help="Source dataset.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers (threads) per process.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size per process.")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation size as portion of the whole train set.")
    
    # Pruning
    parser.add_argument("--mask-gradients", type=int2bool, default=0,
                        help="Perform gradient masking operations.")
    parser.add_argument("--topk", type=float, default=0.5,
                        help="Topk percentage of gradients to retain.")
    parser.add_argument("--ignore-zeroes", type=int2bool, default=1,
                        help="Ignore zero activations when evaluating the mean.")
    
    return parser
