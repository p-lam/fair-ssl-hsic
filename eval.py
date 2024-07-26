import argparse
import os, sys
import torch
import pandas as pd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import wandb

from copy import deepcopy
from math import ceil
from tqdm import tqdm
from utils import *
from modules.model import SSL_HSIC, Fair_SSL_HSIC, Model
from simclr.simclr import SimCLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.linear_model import LogisticRegression
from datasets.cmnist import ColoredMNIST
from datasets.celeba import get_celeba_dataset

# locating dataset folder(s)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# for reproducibility
torch.manual_seed(0)
np.random.seed(0)

models = {"simclr": SimCLR, 
          "ssl-hsic": SSL_HSIC, 
          "fair-ssl-hsic": Fair_SSL_HSIC}

def parse_args():
    parser = argparse.ArgumentParser(description='Train Fair SSL-HSIC')
    # dataset arguments
    parser.add_argument('--dataset', dest='dataset', default='cmnist', type=str, help='dataset (options: cmnist, celeba, dsprites)', choices=['cmnist', 'celeba', 'dsprites'])    
    parser.add_argument("--data", help='path for loading dccata', default='data', type=str)
    # model arguments
    parser.add_argument("--num_workers", help="number of workers", default=4, type=int)
    parser.add_argument('--model', default='simclr', type=str, help='Model to use', choices=['simclr','ssl-hsic','fair-ssl-hsic', 'supervised'])
    parser.add_argument('-a', '--arch', default='resnet18', help='resnet architecture')
    # training args/hyperparameters
    parser.add_argument('--lr', '--learning-rate', default=0.3, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', default=1e-6, type=float, metavar='W', help='weight decay')
    parser.add_argument('--feature_dim', default=64, type=int, help='Feature dim for latent vector')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--n_views', default=2, type=int, help="number of augmentations for simclr/ssl") 
    parser.add_argument('--small_net', action='store_true')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    # misc arguments
    parser.add_argument('--results-dir', default='./results', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')   
    parser.add_argument('--fp16_precision', action='store_true') 
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str) 
    parser.add_argument("--wandb_name", default="eval")
    # parse and return args
    args = parser.parse_args()
    return args

def main(config=None):
    args = parse_args()
    # make sure we are using gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    args.wandb_name = f"{args.wandb_name}_lr{args.lr}" # for sweep tracking

    # setup tracking in wandb
    args_dict = deepcopy(vars(args)) 
    print(f"Saving to wandb under run name: {args.wandb_name}")
    wandb.init(
        project="Eval",
        name=f"{args.wandb_name}" if args.wandb_name is not None else None,
        config=args_dict
    )

    # load and transform data
    if args.dataset == "celeba":
        train_dataset, val_dataset, test_dataset = get_celeba_dataset(n_views=args.n_views)
        args.feature_dim = [256, 128]
    elif args.dataset == "cmnist": 
        dataset_type = ColoredMNIST 
        train_dataset = dataset_type(root='data', env='train', n_views=args.n_views)
        # val_dataset = dataset_type(root='data', env='val', n_views=1)
        test_dataset = dataset_type(root='data', env='test', n_views=1) 
    elif args.dataset == "dsprites":
        pass 
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) 

    # load model
    model = Model(feature_dim=args.feature_dim).to(args.device)

    with wandb.init(config=config):
        config = wandb.config 
        # define optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=args.wd, momentum=0.9)    
        # define cosine scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
        # define loss criterion
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        # load model if needed 
        if args.resume != '':
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print('Loaded from: {}'.format(args.resume))
        # logging
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        # training loop 
        if args.model != "supervised":
            net = models[args.model](model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            top1_test, top5_test = net.evaluate(train_loader, test_loader, epoch=args.epochs - 1)
            wandb.log({"test_top1_acc": top1_test.item(), "test_top5_acc": top5_test.item()})
            print(f"[Final Test Acc@1|5 {top1_test.item():2f}|{top5_test.item():2f}]")
        else:
            # supervised baseline
            print("Not implemented!")

if __name__ == '__main__':
    main()