import os
import torch as tc
import pickle

from torchvision.transforms import functional as F
from torch.utils.data import Dataset

from PIL import Image

class TinyImages(Dataset):
    DATA_FILENAME = 'ti_500K_pseudo_labeled.pickle'
    
    def __init__(self, root, transform=None, target_transform=None, download=False):
        super().__init__()

        data_path = os.path.join(root, self.DATA_FILENAME)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.data = data['data']
        self.targets = data['extrapolated_targets']

        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        aug = img if self.transform is None else self.transform(img)
        target = target if self.target_transform is None else self.target_transform(target)
        
        return aug, target
