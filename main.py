import argparse
import os, sys
import torch
import pandas as pd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp

from pathlib import Path
from math import ceil
from dataloader import get_celeba_dataset, ColoredMNIST
from model import Model
from utils import hsic_inter, hsic_intra
from hsic import HSICLoss
from tqdm import tqdm
from PIL import Image
from skimage import io, transform
from torchvision import transforms, utils, datasets
from torchvision.models.resnet import resnet50, resnet18


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def l2_norm(x):
    return torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12, out=None)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available()==True else 'cpu'
    device = torch.device(device)
    print(f'We are using device name "{device}"')
    parser = argparse.ArgumentParser(description='Dataloader Test')
    parser.add_argument('--feature_dim', default=64, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gamma', default=3, type=int, help='Regularization Coefficient (3 in original impl)')
    parser.add_argument('--dataset', dest='dataset', default='cifar10', type=str, help='dataset (options: cmnist, celeba, cifar10)')    
    parser.add_argument("--data", help='path for loading data', default='data', type=str)
    parser.add_argument("--percent", help="percentage of conflict", default= "1pct", type=str)
    parser.add_argument("--num_workers", help="number of workers", default=4, type=int)

    args = parser.parse_args()

    if args.dataset == "celeba":
        # load dataset
        dataset = get_celeba_dataset()
        train_size = ceil(0.70 * len(dataset))
        val_size = ceil(0.10 * len(dataset))
        test_size = len(dataset) - (train_size+val_size)
        train, test_dataset = torch.utils.data.random_split(dataset, [train_size+val_size, test_size])
        train_dataset, val_dataset = torch.utils.data.random_split(train, [train_size, val_size])
    elif args.dataset == "cmnist": 
        dataset = ColoredMNIST(root='data', env='train')
        train_size = ceil(0.80 * len(dataset))
        val_size = ceil(0.20 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        test_dataset = ColoredMNIST(root='data', env='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # load model
    model = Model(feature_dim=args.feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

    criterion = HSICLoss(num_rff_features=64, regul_weight=args.gamma)

    # train model
    for e in range(0, args.epochs):
        for pos_1, pos_2, label, sen_attr, dataset_idx in train_loader:
            pos_1, pos_2, label = pos_1.to(device), pos_2.to(device), label.to(device)
            batch_pos = torch.eye(pos_1.shape[0]).to(device)
            feat_1, proj_1 = model(pos_1)
            feat_2, proj_2 = model(pos_2)
            # Loss = HSIC(\phi(Z_1), \phi(Z_2)) + \gamma \sqrt(HSIC(\phi(Z_1), \phi(Z_1)))
            feat_1, proj_1 = l2_norm(feat_1), l2_norm(proj_1)
            feat_2, proj_2 = l2_norm(feat_2), l2_norm(proj_2)
            hiddens = [feat_1, feat_2, proj_1, proj_2]
            loss_hsic, summaries = criterion(hiddens)

            optimizer.zero_grad()
            loss_hsic.backward()
            optimizer.step()

            # loss_hsic += hsic_intra(proj_1, proj_2)
            # loss_hsic += args.gamma * hsic_inter(proj_1, proj_2)



# Representation loss.
# hiddens = [online_network_out['prediction_view1'],
#            online_network_out['prediction_view2'],
#            jax.lax.stop_gradient(target_network_out['projection_view2']),
#            jax.lax.stop_gradient(target_network_out['projection_view1'])]
# hiddens = [byol_helpers.l2_normalize(h, axis=-1) for h in hiddens]
# if jax.device_count() > 1:
#   feature_dim = hiddens[0].shape[-1]
#   hiddens = [
#       helpers.all_gather(h).reshape(-1, feature_dim) for h in hiddens
#   ]
# hsic_loss, summaries = self.kernel_loss_fn.apply(kernel_params, rng_hsic,
#                                                  hiddens, rff_kwargs)