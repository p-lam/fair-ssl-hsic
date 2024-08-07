import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from PIL import Image
from torchvision import datasets
from torchvision import transforms as t
from math import ceil

# gpu or cpu
device = 'cuda' if torch.cuda.is_available()==True else 'cpu'
device = torch.device(device)

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

def multicolor_grayscale_arr(img, target=None):
    """Code referenced from https://github.com/NeurAI-Lab/InBiaseD/blob/main/data/data_wrapper.py"""
    dtype = img.dtype
    # color grayscale img based on label 
    if target is not None:
        fg_color = ColoredMNIST.COLORS[target]
    else: # random color 
        rand_target = random.randint(0,9)
        fg_color = ColoredMNIST.COLORS[rand_target]
    img = torch.Tensor(img)
    color_img = img.unsqueeze(dim=-1).repeat(1, 1, 3).float()
    img[img < 75] = 0.0
    img[img >= 75] = 255.0
    color_img /= 255.0
    color_rgb = ColoredMNIST.CHMAP[fg_color]
    color_img[img != 0] = color_rgb
    color_img[img == 0] *= torch.tensor([0, 0, 0])
    return (color_img.numpy()).astype(dtype), color_rgb 

class ColoredMNIST(datasets.VisionDataset):
    """
    ColoredMNIST wrapper
    Training set has fixed colors per digit, test set has randomized colors.
    Modified from https://arxiv.org/pdf/1907.02893.pdf
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
    

    def __init__(self, root='data', env='train', uniform=True, n_views=2, color_jitter=False, s=1):
        super(ColoredMNIST, self).__init__(root,)
        self.prepare_colored_mnist(uniform=uniform)
        test_transform = t.Compose([    t.ToTensor(),
                                        t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if color_jitter: 
            color_jitter = t.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            train_transform =  t.Compose([  t.RandomResizedCrop(size=28, scale=(0.8, 1.)),
                                            t.RandomRotation(15),
                                            t.RandomApply([color_jitter], p=0.8),
                                            t.RandomGrayscale(p=0.2),
                                            t.ToTensor(),
                                            t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])
        else: 
            train_transform =  t.Compose([  t.RandomResizedCrop(size=28, scale=(0.8, 1.)),
                                            t.RandomRotation(15),
                                            t.ToTensor(),
                                            t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])
        self.transform = train_transform if env == "train" else test_transform 
        self.n_views = n_views
        
        if env in ['train','test','val']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train, test')

    def __getitem__(self, index):
        # load the data
        img, label, color = self.data_label_tuples[index]
        imgs = [self.transform(img) for _ in range(self.n_views)]
        if self.n_views == 1:
            imgs = imgs[0]
        return imgs, label, color

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self, uniform=True):
        # get data directory
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')

        # check if cmnist already exists
        if os.path.exists(os.path.join(colored_mnist_dir, 'train.pt')) \
            and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
            print('Colored MNIST dataset already exists')
            return
        
        # else, prepare cmnist 
        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
        val_size = ceil(0.16 * len(train_mnist))
        train_size = len(train_mnist) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_mnist, [train_size, val_size])
        test_dataset = datasets.MNIST(self.root, train=False)
        train_set, val_set, test_set = [], [], []

        # build train dataset
        for idx, (im, label) in enumerate(train_dataset):
            im_array = np.array(im)
            if uniform: 
                colored_arr, color = multicolor_grayscale_arr(im_array, target=label)
                train_set.append([Image.fromarray(colored_arr), label, color / 255.0 ])
            else: 
                if np.random.uniform() < 0.3:
                    colored_arr, color = multicolor_grayscale_arr(im_array, target=label)
                    train_set.append([Image.fromarray(colored_arr), label, color / 255.0 ])
                else: 
                    colored_arr, color = multicolor_grayscale_arr(im_array, target=None)
                    train_set.append([Image.fromarray(colored_arr), label, color / 255.0])
        
        # build val and test datasets
        for idx, (im, label) in enumerate(val_dataset): 
            im_array = np.array(im)
            colored_arr, color = multicolor_grayscale_arr(im_array, target=None)
            val_set.append([Image.fromarray(colored_arr), label, color / 255.0])

        for idx, (im, label) in enumerate(test_dataset): 
            im_array = np.array(im)
            colored_arr, color = multicolor_grayscale_arr(im_array, target=None)
            test_set.append([Image.fromarray(colored_arr), label, color / 255.0])

        # save new data
        os.makedirs(colored_mnist_dir)
        torch.save(train_set, os.path.join(colored_mnist_dir, 'train.pt'))
        torch.save(val_set, os.path.join(colored_mnist_dir, 'val.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))

def plot_dataset_digits(dataset, train=True):
    """
    Plot colored mnist (train or test), with three lines of digits for the training set and one for the test set
    """
    label_to_images = {i: [] for i in range(10)}  # dictionary to store images by label
    columns = 10 
    rows = 3 if train else 1 
    fig, axes = plt.subplots(rows, columns, figsize=(columns,rows), gridspec_kw = {'wspace':0.05, 'hspace':0.05}) 
    # iterate through dataset 
    for i in range(len(dataset)):
        img, label, color = dataset[i]
        if label in label_to_images and len(label_to_images[label]) < 3:  # ensure 3 images per label
            label_to_images[label].append((img, label, color))
        if all(len(images) == rows for images in label_to_images.values()):
            break  # stop if we have 3 images for each label
    # Plot images
    for col in range(columns):
        for row in range(rows):
            img, label, color = label_to_images[col][row]
            ax = axes[row, col] if rows == 3 else axes[col]
            img = img.permute(1, 2, 0)
            ax.imshow(img, cmap='gray')  # Use cmap='gray' for MNIST images
            ax.axis('off')
            if row == 0:  # Only set title for the top row
                ax.set_title(f"Label: {label}")
    
    name = "mnist_train" if train else "mnist_test"
    plt.savefig(name, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    # test if dataset code is working / plot datasets
    train_set = ColoredMNIST(root='data', env='train', n_views=1)
    val_set = ColoredMNIST(root='data', env='val', n_views=1)
    test_set = ColoredMNIST(root='data', env='test', n_views=1)
    plot_dataset_digits(train_set)
    plot_dataset_digits(val_set, train=False)
    plot_dataset_digits(test_set, train=False)
    # plot_dataset_digits(test_set, train=False)