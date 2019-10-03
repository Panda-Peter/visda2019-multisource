import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

class DistributedSamplerOnline(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=None, index=None, distributed=False):
        if num_replicas is None:
            if not distributed:
                num_replicas = 1
            else:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
   
        if rank is None:
            if not distributed:
                rank = 0
            else:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()

        self.epoch = 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.index = index
        self.num_samples = int(math.ceil(len(self.index) * 1.0 / self.num_replicas))
        
    def __iter__(self):
        indices = []
        for i in range(self.rank * self.batch_size, len(self.index), self.num_replicas * self.batch_size):
            if i + self.batch_size <= len(self.index):
                indices += self.index[i:i + self.batch_size]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch