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
                                t.RandomResizedCrop(64),
                                t.RandomHorizontalFlip(p=0.5),
                                t.RandomApply([t.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                t.RandomGrayscale(p=0.2)])
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
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        img = img.resize((64, 64))
        # Apply transformations to the image
        imgs = [self.transform(img) for _ in range(self.n_views)]
        if self.n_views == 1:
            imgs = imgs[0]
        label = np.array(self.label.iloc[idx][1:])[20]  # male
        sen_attr = np.array(self.label.iloc[idx][1:])[-1]  # young
        return imgs, label, sen_attr

def get_celeba_dataset(batch_size, n_views=1):
    root = Path(DATA_ROOT)/"KaggleCeleb"
    anno_path = root / "list_attr_celeba.csv"
    dataset = CelebADataset(root, anno_path, n_views=n_views)
    train_size = ceil(0.64 * len(dataset))
    val_size = ceil(0.16 * len(dataset))
    test_size = len(dataset) - (train_size+val_size)
    train_, test_dataset = torch.utils.data.random_split(dataset, [train_size+val_size, test_size])
    train_dataset, val_dataset = torch.utils.data.random_split(train_, [train_size, val_size])
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataloader Test')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    args = parser.parse_args()
    # test if dataset code is working / plot datasets
    train_dataset, val_dataset, test_dataset = get_celeba_dataset(args.batch_size, n_views=2)
    print("complete")