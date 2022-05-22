import torch


class AverageMeter:
    def __init__(self) -> None:
        self.avg = None
        self.sum = None
        self.count = None
    
    def __init_counters(self, length):
        if length == 0:
            self.avg = 0
            self.sum = 0
            self.count = 0
        else:
            self.avg = []
            self.sum = []
            self.count = 0
            for i in range(length):
                self.avg.append(0)
                self.sum.append(0)
    
    def update(self, val, n):
        if isinstance(val, list):
            if self.avg is None or self.sum is None or self.count is None:
                self.__init_counters(len(val))
            self.sum = [x + y for x, y in zip(self.sum, [x * n for x in val])]
            self.count += n
            self.avg = [x / self.count for x in self.sum]
        else:
            if self.avg is None or self.sum is None or self.count is None:
                self.__init_counters(0)
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
    def value(self):
        return self.avg


class RollingMeter:
    def __init__(self, config) -> None:
        self.outputs = torch.tensor([], device=config.device)
        self.targets = torch.tensor([], device=config.device)
    
    def update(self, outputs, targets):
        self.outputs = torch.cat((self.outputs, outputs), dim=0)
        self.targets = torch.cat((self.targets, targets), dim=0)
        return self.outputs, self.targets
    
    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)
    
    def value(self):
        raise NotImplementedError


class Accuracy(RollingMeter):
    """Evaluates the predictions accuracy given an output `torch.Tensor` and a target `torch.Tensor`.
    Args:
        topk (Tuple[int, ...], optional): top-k accuracy identifiers. E.g. to evaluate both top-1 and top-5 accuracy `topk = (1, 5)`.
    """
    
    def __init__(self, config, topk=(1,)):
        super().__init__(config)
        self.topk = topk
    
    def value(self) -> list:
        """Evaluates the accuracy of the outputs given the targets.
        Returns:
            list: list of top-k accuracy, one for each element of `topk`.
        """
        outputs, targets = self.outputs, self.targets
        
        maxk = max(self.topk)
        batch_size = targets.shape[0]
        if batch_size == 0:
            raise RuntimeError('Warning: batch_size is 0')
        
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1. / batch_size).item())
        
        if len(res) == 1:
            res = res[0]
        
        return res * 100
