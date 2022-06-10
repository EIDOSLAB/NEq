from experiments.resnet.cifar10 import CIFAR10_Freeze_Bprop_Random_Constant, CIFAR10_Freeze_Bprop_Bottomk_Constant, \
    CIFAR10_NEq
from experiments.templates.resnet import CIFAR10_Base

__all__ = [CIFAR10_Base, CIFAR10_Freeze_Bprop_Random_Constant, CIFAR10_Freeze_Bprop_Bottomk_Constant, CIFAR10_NEq]


def get_experiment_by_name(name):
    for Experiment in __all__:
        if Experiment.__name__ == name:
            return Experiment
