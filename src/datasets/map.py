from torch.utils.data import Dataset


class MapDataset(Dataset):
    def __init__(self, dataset, map_fn, with_target=False):
        self.dataset = dataset
        self.map = map_fn
        self.with_target = with_target
    
    def __getitem__(self, index):
        if self.with_target:
            return self.map(self.dataset[index][0], self.dataset[index][1])
        else:
            return self.map(self.dataset[index][0]), self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)
