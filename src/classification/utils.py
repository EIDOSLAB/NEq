from torch.optim.lr_scheduler import MultiStepLR, StepLR

from optim import MaskedSGD, MaskedAdam


def get_optimizer(config, model):
    print(f"Initialize optimizer {config.optim}")
    
    # Define optimizer and scheduler
    named_params = list(map(list, zip(*list(model.named_parameters()))))
    if config.optim == "sgd":
        return MaskedSGD(named_params[1], names=named_params[0], lr=config.lr, weight_decay=config.weight_decay,
                         momentum=config.momentum)
    if config.optim == "adam":
        return MaskedAdam(named_params[1], names=named_params[0], lr=config.lr, weight_decay=config.weight_decay)


def get_scheduler(config, optimizer):
    print("Initialize scheduler")
    
    if config.dataset == "cifar10":
        return MultiStepLR(optimizer, milestones=[100, 150])
    if config.dataset == "imagenet":
        return StepLR(optimizer, step_size=30)
