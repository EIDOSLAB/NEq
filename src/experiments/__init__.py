from experiments.classification.cifar10 import CIFAR10_NEq
from experiments.templates.cifar10 import CIFAR10_Base

__all__ = [CIFAR10_Base, CIFAR10_NEq]


def get_experiment_by_name(name):
    for Experiment in __all__:
        if Experiment.__name__ == name:
            return Experiment
