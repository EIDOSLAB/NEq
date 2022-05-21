from experiments.classification.cifar import CIFAR10_Base, CIFAR10_NEq

__all__ = [CIFAR10_Base, CIFAR10_NEq]


def get_experiment_by_name(name):
    for Experiment in __all__:
        if Experiment.__name__ == name:
            return Experiment
