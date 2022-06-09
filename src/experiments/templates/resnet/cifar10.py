import wandb
from torch import nn
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
        self.project_name = 'NEq'
        self.tag = "cifar10-base"
    
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
            return SGD(self.model.parameters(), lr=self.opts.lr,
                       weight_decay=self.opts.weight_decay, momentum=self.opts.momentum)
        if self.opts.optimizer == "adam":
            return Adam(self.model.parameters(), lr=self.opts.lr,
                        weight_decay=self.opts.weight_decay)
    
    def load_scheduler(self):
        return MultiStepLR(self.optimizer, milestones=[100, 150])


class CIFAR10_Freeze_Bprop_Base(CIFAR10_Base):
    __help__ = "Template for Freeze Back-propagation experiments on CIFAR-10"
    
    def __init__(self, opts):
        super().__init__(opts)
        self.masks = {}
        self.project_name = 'NEq'
        self.tag = "freeze-bprop-base"
    
    @staticmethod
    def load_config(parser):
        CIFAR10_Base.load_config(parser)
        parser.add_argument('--warmup', type=int, help='number of warmup epochs', default=1)
    
    def load_optimizer(self):
        named_params = list(map(list, zip(*list(self.model.named_parameters()))))
        if self.opts.optimizer == "sgd":
            return MaskedSGD(named_params[1], names=named_params[0], lr=self.opts.lr,
                             weight_decay=self.opts.weight_decay, momentum=self.opts.momentum)
        if self.opts.optimizer == "adam":
            return MaskedAdam(named_params[1], names=named_params[0], lr=self.opts.lr,
                              weight_decay=self.opts.weight_decay)
    
    def update_masks(self):
        self.masks = {}
        
        for group in self.optimizer.param_groups:
            group["masks"] = self.masks
    
    def log_masks(self, epoch):
        neq_neurons = {}
        
        for n, m in self.model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                neq_neurons[f"{n}"] = (self.masks[n].shape[0] / m.weight.shape[0]) * 100
        
        wandb.log({
            "neq-neurons": neq_neurons,
            "total":       (sum(neq_neurons.values()) / self.total_neurons) * 100
        }, step=epoch)
    
    def compute_masks(self, epoch):
        self.update_masks()
