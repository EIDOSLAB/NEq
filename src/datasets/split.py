import numpy as np
from torch import Generator
from torch.utils.data import random_split


def split_dataset(dataset, percentage, random_seed):
    dataset_length = len(dataset)
    valid_length = int(np.floor(percentage * dataset_length))
    train_length = dataset_length - valid_length
    
    # return order: tran_dataset, validation_dataset
    return random_split(dataset, [train_length, valid_length], Generator().manual_seed(random_seed))
