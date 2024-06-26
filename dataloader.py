import argparse
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
import random

from tqdm import tqdm
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms as t
from pathlib import Path
from math import ceil

# gpu or cpu
device = 'cuda' if torch.cuda.is_available()==True else 'cpu'
device = torch.device(device)

# point $DATA_ROOT to location of CelebA folder
DATA_ROOT = os.environ["DATA_ROOT"]

"""
CelebA helpers
"""
train_transform = t.Compose([
                    t.RandomResizedCrop(64),
                    t.RandomHorizontalFlip(p=0.5),
                    t.RandomApply([t.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    t.RandomGrayscale(p=0.2)])

trans = t.Compose([ t.ToTensor(),
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

"""
Colored MNIST helpers
"""
def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])

    if red:
        arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype), arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)
    
    return arr

def multicolor_grayscale_arr(img):
    """Code referenced from https://github.com/NeurAI-Lab/InBiaseD/blob/main/data/data_wrapper.py"""
    # if self.train:
    #     fg_color = ColoredMNIST.COLORS[target]
    dtype = img.dtype
    rand_target = random.randint(0,9)
    fg_color = ColoredMNIST.COLORS[rand_target]
    img = torch.Tensor(img)
    color_img = img.unsqueeze(dim=-1).repeat(1, 1, 3).float()
    img[img < 75] = 0.0
    # img[img >= 75] = 255.0
    # color_img /= 255.0
    color_img[img != 0] = ColoredMNIST.CHMAP[fg_color]
    color_img[img == 0] *= torch.tensor([0, 0, 0])

    return (color_img.numpy()).astype(dtype)

class ColoredMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """
    COLORS = ['red', 'orange', 'yellow', 'greenyellow', 'green', 'cyan', 'blue', 'purple', 'pink', 'magenta']
    CHMAP = {
        "red": (torch.tensor([255.0, 0, 0])),
        "orange": (torch.tensor([255.0, 165.0, 0])),
        "yellow": (torch.tensor([255.0, 255.0, 0])),
        "greenyellow": (torch.tensor([173.0, 255.0, 47.0])),
        "green": (torch.tensor([0, 255.0, 0])),
        "cyan": (torch.tensor([0, 255.0, 255.0])),
        "blue": (torch.tensor([0, 0, 255.0])),
        "purple": (torch.tensor([160.0, 32.0, 240.0])),
        "pink": (torch.tensor([219.0, 11.0, 117.0])),
        "magenta": (torch.tensor([255.0, 0, 255.0]))
    }

    def __init__(self, root='data', env='train', transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.transform = trans
        self.train_transform =  t.Compose([
                                t.RandomCrop(24),
                                t.RandomHorizontalFlip(p=0.5),
                                t.ToTensor(),
                                ])
        self.prepare_colored_mnist()
        if env in ['train','test']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train, test')

    def __getitem__(self, index):
        # load the data
        img, label, color = self.data_label_tuples[index]
        # transform the images
        pos_1 = self.train_transform(img)
        pos_2 = self.train_transform(img)
        return pos_1, pos_2, label, color, index 

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        if os.path.exists(os.path.join(colored_mnist_dir, 'train.pt')) \
            and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
            print('Colored MNIST dataset already exists')
            return
        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
        train_set, test_set = [], []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with 25% probability
            if np.random.uniform() < 0.25:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability
            if idx < 40000:
                if np.random.uniform() < 0.2:
                    color_red = not color_red
            else: # this isn't used
                if np.random.uniform() < 0.9:
                    color_red = not color_red

            if idx < 40000:
                colored_arr = color_grayscale_arr(im_array, red=color_red)
                train_set.append([Image.fromarray(colored_arr), label, int(color_red)])
            else:
                colored_arr = multicolor_grayscale_arr(im_array)
                test_set.append([Image.fromarray(colored_arr), label, int(color_red)])

        # save new data
        os.makedirs(colored_mnist_dir)
        torch.save(train_set, os.path.join(colored_mnist_dir, 'train.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))

def plot_dataset_digits(dataset, train=True):
    fig = plt.figure(figsize=(13, 8))
    columns = 6
    rows = 3
    ax = []

    for i in range(columns * rows):
        img, label, color = dataset[i]
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        if train:
            ax[-1].set_title("Label: " + str(label) + ", Color: " + str(color))  # set title
        else: 
            ax[-1].set_title("Label: " + str(label))
        plt.imshow(img)

    plt.show()  # finally, render the plot
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataloader Test')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', dest='dataset', default='cifar10', type=str, help='dataset (options: cmnist, celeba, cifar10)')    

    # args parse
    args = parser.parse_args()
    if args.dataset == "celeba":
        celeba = get_celeba_dataset()
        train_size = ceil(0.70 * len(celeba))
        val_size = ceil(0.10 * len(celeba))
        test_size = len(celeba) - (train_size+val_size)
        train, test_dataset = torch.utils.data.random_split(celeba, [train_size+val_size, test_size])
        train_dataset, val_dataset = torch.utils.data.random_split(train, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2*args.batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=False)
    else: 
        train_set = ColoredMNIST(root='data', env='test')
        plot_dataset_digits(train_set, train=False)