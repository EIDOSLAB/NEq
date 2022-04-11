from typing import Union, Tuple, Optional

import torch


class AverageMeter:
    """
    Computes and stores the average of a given value(s). The average value is store in the `avg` parameter.

    Examples:
            ```python
            # Initialize a meter to record loss
            losses = AverageMeter()
            # Update meter after every minibatch update
            losses.update(loss_value, batch_size)
            average_loss = losses.avg
    """
    
    def __init__(self) -> None:
        self.avg = None
        self.sum = None
        self.count = None
    
    def __init_counters(self, length: int) -> None:
        """Initializes the `AverageMeter` counters.

        Args:
            length (int): number of elements to

        Returns:
            None:
        """
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
    
    def update(self, val: Union[int, float, list], n: int) -> None:
        """Updates `avg`.

        Args:
            val (Union[int, float, list]): value used to update `avg`.
            n (int): identifies the number of times `val` must be repeated, e.g. how many batches had been used to define the `val`.

        Returns:
            None:
        """
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


class Accuracy:
    """Evaluates the predictions accuracy given an output `torch.Tensor` and a target `torch.Tensor`.

    Args:
        topk (Tuple[int, ...], optional): top-k accuracy identifiers. E.g. to evaluate both top-1 and top-5 accuracy `topk = (1, 5)`.
    """
    
    def __init__(self, topk: Optional[Tuple[int, ...]] = (1,)) -> None:
        self.topk = topk
    
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> list:
        """Evaluates the accuracy of the outputs given the targets.

        Args:
            outputs (torch.Tensor): tensor defining a prediction.
            targets (torch.Tensor): tensor defining the targets.

        Returns:
            list: list of top-k accuracy, one for each element of `topk`.

        """
        maxk = max(self.topk)
        batch_size = targets.shape[0]
        
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        if len(res) == 1:
            res = res[0]
        return res
