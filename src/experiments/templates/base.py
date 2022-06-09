"""
Base class for all experiments.
Provides common interface for loading data,
models, loss functions and fitting/testing
"""
from abc import ABCMeta, abstractmethod

import torch.cuda


class ExperimentBase(metaclass=ABCMeta):
    def __init__(self, opts):
        self.opts = opts
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.opts.device = self.device
        print('Creating experiment', self.__class__.__name__)
        print('Options: (')
        for k, v in vars(opts).items():
            print(f'\t{k}: {v}')
        print(')')
    
    @staticmethod
    def load_config(parser):
        raise NotImplementedError
    
    @abstractmethod
    def initialize(self):
        raise NotImplementedError
    
    @abstractmethod
    def run(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_data(self):
        raise NotImplementedError


if __name__ == '__main__':
    experiment = ExperimentBase(None)
    experiment.initialize()
