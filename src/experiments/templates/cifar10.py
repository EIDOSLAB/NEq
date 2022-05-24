from abc import abstractmethod

from torch.nn.functional import cross_entropy
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from datasets import get_dataloaders
from experiments.templates.classification import ClassificationLearningExperiment
from metrics import Accuracy
from models import get_model
from optimizers import MaskedSGD, MaskedAdam


class CIFAR10_Base(ClassificationLearningExperiment):
    __help__ = "Template learning experiments on CIFAR-10"
    
    def __init__(self, opts):
        super().__init__(opts)
        self.validation = None
    
    def get_experiment_key(self):
        return ""
    
    def compute_loss(self, outputs, minibatch):
        return {"ce": cross_entropy(outputs, minibatch[1])}
    
    def init_meters(self):
        return {"accuracy.top1": Accuracy(topk=(1,), config=self.opts)}
    
    def update_meters(self, meters, outputs, minibatch):
        meters["accuracy.top1"].update(outputs, minibatch[1])
    
    def optimize_metric(self):
        return None
    
    @staticmethod
    def load_config(parser):
        ClassificationLearningExperiment.load_config(parser)
        parser.add_argument('--model', type=str, help='resnet variant',
                            choices=['resnet32-cifar10', 'resnet50-cifar10', 'resnet56-cifar10', 'resnet110-cifar10'],
                            default='resnet32-cifar10')
    
    def load_data(self):
        dataloaders = get_dataloaders(self.opts)
        self.validation = dataloaders["validation"]
        return {"train": dataloaders["train"], "test": dataloaders["test"]}
    
    def load_model(self):
        model, total_neurons = get_model(self.opts)
        
        return model, total_neurons
    
    def load_optimizer(self):
        if self.opts.optimizer == "sgd":
            optimizer = SGD(self.model.parameters(), lr=self.opts.lr,
                            weight_decay=self.opts.weight_decay, momentum=self.opts.momentum)
        if self.opts.optimizer == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.opts.lr,
                             weight_decay=self.opts.weight_decay)
        
        scheduler = MultiStepLR(optimizer, milestones=[100, 150])
        
        return optimizer, scheduler


class CIFAR10_Freeze_Bprop_Base(CIFAR10_Base):
    __help__ = "Template for Freeze Back-propagation experiments on CIFAR-10"
    
    def __init__(self, opts):
        super().__init__(opts)
        self.masks = {}
    
    def load_optimizer(self):
        named_params = list(map(list, zip(*list(self.model.named_parameters()))))
        if self.opts.optimizer == "sgd":
            optimizer = MaskedSGD(named_params[1], names=named_params[0], lr=self.opts.lr,
                                  weight_decay=self.opts.weight_decay, momentum=self.opts.momentum)
        if self.opts.optimizer == "adam":
            optimizer = MaskedAdam(named_params[1], names=named_params[0], lr=self.opts.lr,
                                   weight_decay=self.opts.weight_decay)
        
        scheduler = MultiStepLR(optimizer, milestones=[100, 150])
        
        return optimizer, scheduler
    
    def __update_masks(self):
        self.masks = {}
        
        for group in self.optimizer.param_groups:
            group["masks"] = self.masks
    
    @abstractmethod
    def __compute_masks(self, epoch):
        raise NotImplementedError
