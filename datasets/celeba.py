import argparse
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
import random
import matplotlib.gridspec as gridspec
import csv 

from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as t
from torchvision import datasets
from pathlib import Path
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# point $DATA_ROOT to location of CelebA folder
DATA_ROOT = os.environ["DATA_ROOT"]

"""
CelebA helpers
"""
class CelebADataset(Dataset):
    def __init__(self, root_dir=Path(DATA_ROOT)/"CelebA", n_views=1, split='Train', targets=[], sensitives=[]):
        """
        CelebA wrapper
        Args:
        root_dir (string): Directory with all the images
        n_views: Number of contrastive views
        split: Train, val, or test
        """
        # Read names of images in the root directory
        self.root_dir = root_dir
        self.img_dir = root_dir / "img_align_celeba" / "img_align_celeba"

        if split == 'train':
            self.transform = t.Compose([
                            t.RandomResizedCrop(size=64),
                            t.RandomHorizontalFlip(p=0.5),
                            t.RandomApply([t.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            t.RandomGrayscale(p=0.2),
                            t.ToTensor(), 
                            t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.n_views = n_views
        elif split in ['test','val']: 
            self.transform = t.Compose([t.ToTensor(),
                                        t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.n_views = 1
        else:
            print("Invalid split!")
        
        self.split = split
        self.n_views = n_views
        self.labels = pd.read_csv(root_dir / "list_attr_celeba.csv")
        self.splits = pd.read_csv(root_dir / "list_eval_partition.csv")
        self.image_names = self.get_img_split()

        self.targets = targets 
        self.sens_atts = sensitives

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the path to the image 
        img_path = os.path.join(self.img_dir, self.image_names[idx])

        # Get the associated labels
        label = np.array(self.labels.iloc[idx][1:][self.targets[0]])  # male
        sen_attr = np.array(self.labels.iloc[idx][1:][self.sens_atts[0]])  # young

        if len(self.targets) > 1:
            label2 = np.array(self.labels.iloc[idx][1:][self.targets[1]])
            label = label + 2*label2
        
        if len(self.sens_atts) > 1:
            sen_attr2 = np.array(self.labels.iloc[idx][1:][self.sens_att[1]])
            sen_attr = sen_attr + 2 * sen_attr2 

        # shift binary labels from (1, -1) to (1, 0)
        sen_attr = np.where(sen_attr == 1, 1, 0)  
        label = np.where(label == 1, 1, 0)

        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        img = img.resize((128, 128))
        imgs = [self.transform(img) for _ in range(self.n_views)]
        if self.n_views == 1:
            imgs = imgs[0]

        return imgs, label, sen_attr

    def get_img_split(self): # we follow the official train/val/test split
        if self.split == 'train':
            imgs = self.splits[self.splits['partition'] == 0]['image_id'].tolist()    
        elif self.split == 'val':
            imgs = self.splits[self.splits['partition'] == 1]['image_id'].tolist()
        elif self.split == 'test': 
            imgs = self.splits[self.splits['partition'] == 2]['image_id'].tolist()
        else: 
            print("Invalid split!")
        return natsorted(imgs) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataloader Test')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    args = parser.parse_args()
    # test if dataset code is working / plot datasets
    root = Path(DATA_ROOT)/"CelebA"
    train_dataset= CelebADataset(root_dir=root, split="train")
    val_dataset = CelebADataset(root_dir=root, split="val")
    test_dataset = CelebADataset(root_dir=root, split="test")
    print("complete")