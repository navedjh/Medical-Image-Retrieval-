"""
PneumoniaMNIST dataset wrapper
"""

import medmnist
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class PneumoniaMNISTDataset(Dataset):
    def __init__(self, split='train'):
        self.dataset = medmnist.PneumoniaMNIST(split=split, download=True)
        self.split = split
        self.label_names = {0: 'Normal', 1: 'Pneumonia'}
        print(f"  Loaded {split} set: {len(self.dataset)} images")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_array = np.array(img)
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0) / 255.0
        return img_tensor, label[0], idx
    
    def get_pil_image(self, idx):
        img, label = self.dataset[idx]
        return img, label[0]

def create_dataloader(dataset, batch_size=64, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)