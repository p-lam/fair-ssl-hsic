import argparse
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
import random
import matplotlib.gridspec as gridspec

from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as t
from pathlib import Path
from math import ceil
from simclr.data_aug import GaussianBlur
from copy import deepcopy
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# point $DATA_ROOT to location of CelebA folder
DATA_ROOT = os.environ["DATA_ROOT"]

"""
CelebA helpers
"""
class CelebADataset(Dataset):
    def __init__(self, root_dir, csv_file, n_views=1, train=True):
        """
        CelebA wrapper

        Args:
        root_dir (string): Directory with all the images
        transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        self.root_dir = root_dir
        self.img_dir = root_dir / "img_align_celeba" / "img_align_celeba"
        self.test_transform = t.Compose([t.ToTensor(),
                                    t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_transform = t.Compose([
                                t.RandomResizedCrop(size=128, scale=(0.2, 1.)),
                                t.RandomHorizontalFlip(),
                                t.RandomApply([t.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                t.RandomGrayscale(p=0.2),
                                t.ToTensor(), 
                                t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image_names = os.listdir(self.img_dir)
        self.n_views = n_views
        self.image_names = natsorted(image_names)
        self.label = pd.read_csv(csv_file)
        self.train = train
    
    def set_transform_mode(self, train=None):
        if train is None:
            train = self.train
        self.transform = self.train_transform if train else self.test_transform
        if not train:
            self.n_views = 1

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the path to the image 
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        # Get the associated labels
        label = np.array(self.label.iloc[idx][1:])[20]  # male
        sen_attr = np.array(self.label.iloc[idx][1:])[-1]  # young
        return img_path, label, sen_attr
    
class CelebASubset(Dataset):
    def __init__(self, subset, n_views=1, augment=True):
        self.dataset = subset
        if augment:
            self.transform = t.Compose([
                            t.RandomResizedCrop(size=128, scale=(0.2, 1.)),
                            t.RandomHorizontalFlip(),
                            t.RandomApply([t.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            t.RandomGrayscale(p=0.2),
                            t.ToTensor(), 
                            t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.n_views = n_views
        else: 
            self.transform = t.Compose([t.ToTensor(),
                                        t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.n_views = 1
    
    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path, label, sen_attr = self.dataset[idx]
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        img = img.resize((128, 128))
        imgs = [self.transform(img) for _ in range(self.n_views)]
        if self.n_views == 1:
            imgs = imgs[0]
        return imgs, label, sen_attr

def get_celeba_dataset(n_views=1):
    root = Path(DATA_ROOT)/"KaggleCeleb"
    anno_path = root / "list_attr_celeba.csv"
    dataset = CelebADataset(root, anno_path, n_views=n_views)
    train_prop = 0.6
    val_prop = 0.2
    test_prop = 0.2
    train_ind, valtest_ind = train_test_split([i for i in range(len(dataset))], test_size=1-train_prop, random_state=1)
    val_ind, test_ind = train_test_split(valtest_ind, test_size=test_prop / (test_prop + val_prop), random_state=1)
    train_dataset = CelebASubset(Subset(dataset, indices=train_ind), n_views=n_views, augment=True)
    val_dataset = CelebASubset(Subset(dataset, indices=val_ind), augment=False)
    test_dataset = CelebASubset(Subset(dataset, indices=test_ind), augment=False)
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataloader Test')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    args = parser.parse_args()
    # test if dataset code is working / plot datasets
    train_dataset, val_dataset, test_dataset = get_celeba_dataset(n_views=2)
    print("complete")