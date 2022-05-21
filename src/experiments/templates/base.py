"""
Base class for all experiments.
Provides common interface for loading data,
models, loss functions and fitting/testing
"""
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from typing import Dict


class ExperimentBase(metaclass=ABCMeta):
    def __init__(self, opts):
        self.opts = opts
        self.device = getattr(opts, 'device', 'cpu')
        print('Creating experiment', self.__class__.__name__)
        print('Options: (')
        for k, v in vars(opts).items():
            print(f'\t{k}: {v}')
        print(')')
    
    @staticmethod
    def load_config(parser, defaults):
        raise NotImplementedError
    
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def load_data(self):
        raise NotImplementedError
    
    @abstractmethod
    def run(self):
        raise NotImplementedError


if __name__ == '__main__':
    experiment = ExperimentBase(None)
    experiment.initialize()
