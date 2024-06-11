import argparse
import os
import torch
from tqdm import tqdm
import pandas as pd
from natsort import natsorted
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, datasets
import torchvision as tv
from torchvision import transforms as t
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from pathlib import Path
from math import ceil
import os 

device = 'cuda' if torch.cuda.is_available()==True else 'cpu'
device = torch.device(device)

# point $DATA_ROOT to location of CelebA folder
DATA_ROOT = os.environ["DATA_ROOT"]

train_transform = t.Compose([
    t.RandomResizedCrop(64),
    t.RandomHorizontalFlip(p=0.5),
    t.RandomApply([t.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    t.RandomGrayscale(p=0.2)])

trans = t.Compose([t.ToTensor(),
                       t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class CelebADataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=trans):
        """
        Args:
        root_dir (string): Directory with all the images
        transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        self.root_dir = root_dir
        self.img_dir = root_dir / "img_align_celeba" / "img_align_celeba"
        self.transform = trans
        image_names = os.listdir(self.img_dir)
        self.image_names = natsorted(image_names)
        self.label = pd.read_csv(csv_file)

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the path to the image 
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        img = img.resize((64, 64))
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        label = np.array(self.label.iloc[idx][1:])[20]  # male
        sen_attr = np.array(self.label.iloc[idx][1:])[-1]  # young
        # Apply transformations to the image
        # img = self.transform(img)
        pos_1 = self.transform(pos_1)
        pos_2 = self.transform(pos_2)

        return pos_1, pos_2, label, sen_attr, idx


def get_celeba_dataset(transform=train_transform):
    root = Path(DATA_ROOT)/"KaggleCeleb"
    anno_path = root / "list_attr_celeba.csv"
    dataset = CelebADataset(root, anno_path, transform=transform)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataloader Test')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    celeba = get_celeba_dataset()
    train_size = ceil(0.70 * len(celeba))
    val_size = ceil(0.10 * len(celeba))
    test_size = len(celeba) - (train_size+val_size)
    train, test_dataset = torch.utils.data.random_split(celeba, [train_size+val_size, test_size])
    train_dataset, val_dataset = torch.utils.data.random_split(train, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2*args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=False)
