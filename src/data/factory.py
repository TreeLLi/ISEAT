import torch as tc
import torchvision.transforms as T
from torch.utils.data import random_split, Subset

from src.utils.printer import dprint
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

DATASETS = {'CIFAR10' : CIFAR10,
            'CIFAR100' : CIFAR100,
            'SVHN' : SVHN}

def fetch_dataset(dataset, root, train=True, input_dim=None, split=False, download=False):    
    assert dataset in DATASETS
    
    # hyper-parameter report
    head = 'Training Set' if train else 'Test Set'
    dprint(head, dataset=dataset)

    if train:
        augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(input_dim[-1], padding=4),
            T.ToTensor()
        ])
    else:
        augment = T.ToTensor()
    
    dataset = DATASETS[dataset](root, train, download=download, transform=augment)

    if split > 1:
        total = len(dataset)
        chunk = total // split
        split = [chunk for i in range(split-1)]
        split = [0] + split + [total-sum(split)]
        split = [sum(split[:i+1]) for i, _ in enumerate(split)]
        indices = list(range(len(dataset)))
        dataset = [Subset(dataset, indices[s:split[i+1]]) for i, s in enumerate(split[:-1])]

    return dataset
