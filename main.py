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
from datasets.celeba import CelebADataset

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
    parser.add_argument('--dataset', dest='dataset', default='cmnist', type=str, help='dataset (options: cmnist, celeba)', choices=['cmnist', 'celeba'])    
    parser.add_argument("--data", help='path for loading data', default='data', type=str)
    # model arguments
    parser.add_argument("--num_workers", help="number of workers", default=4, type=int)
    parser.add_argument('--model', default='simclr', type=str, help='Model to use', choices=['simclr','ssl-hsic','fair-ssl-hsic', 'supervised'])
    parser.add_argument('-a', '--arch', default='resnet18', help='resnet architecture')
    # training args/hyperparameters
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--feature_dim', default=64, type=int, help='Feature dim for latent vector')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--start_epoch', default=1, type=int, help='Starting epoch')
    parser.add_argument('--lamb', default=2, type=float, help='Regularization Coefficient')
    parser.add_argument('--gamma', default=3, type=float, help='Regularization Coefficient')
    parser.add_argument('--hsic_type', default='regular', type=str, help='type of hsic approx: regular, normalized, normalized cca') 
    parser.add_argument('--n_views', default=2, type=int, help="number of augmentations for simclr/ssl") 
    parser.add_argument('--temperature', default=0.07, type=float, help="contrastive temperature") 
    parser.add_argument('--color_jitter', action='store_true')
    parser.add_argument('--small_net', action='store_true')
    parser.add_argument('--targets', default="Attractive")
    parser.add_argument('--sensitives', default="Male")
    # misc arguments
    parser.add_argument('--bn_splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')
    parser.add_argument('--results-dir', default='./results', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')   
    parser.add_argument('--log_every_n_steps', default=10, type=int) 
    parser.add_argument('--fp16_precision', action='store_true') 
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str) 
    parser.add_argument("--wandb_name", default="train")
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
        project="Fair-SSL-HSIC-Sweeps",
        name=f"{args.wandb_name}" if args.wandb_name is not None else None,
        config=args_dict
    )

    # load and transform data
    if args.dataset == "celeba":
        train_dataset= CelebADataset(split="train", n_views=args.n_views, targets=[args.targets], sensitives=[args.sensitives])
        val_dataset = CelebADataset(split="val", n_views=1, targets=[args.targets], sensitives=[args.sensitives])
        test_dataset = CelebADataset(split="test", n_views=1, targets=[args.targets], sensitives=[args.sensitives])
        args.feature_dim = [256, 128]
    elif args.dataset == "cmnist": 
        dataset_type = ColoredMNIST 
        train_dataset = dataset_type(root=args.data, env='train', n_views=args.n_views, color_jitter=args.color_jitter)
        val_dataset = dataset_type(root=args.data, env='val', n_views=1)
        test_dataset = dataset_type(root=args.data, env='test', n_views=1) 
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) 

    # load model
    args.num_classes = 10 if args.dataset == "cmnist" else 2
    model = Model(num_classes=args.num_classes, small=args.small_net, feature_dim=args.feature_dim).to(args.device)

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
            net.train(train_loader, test_loader, val_loader)
        else:
            # supervised baseline
            for epoch in range(args.epochs):
                loss, top1_train, top5_train = train(model, args, epoch, train_loader, criterion, optimizer, scheduler)
                top1_test, top5_test = test(model, args, test_loader)
                # log results
                wandb.log({"train_top1_acc": top1_train.item(), "train_top5_acc":top5_train.item(), 
                            "train_loss": loss.item(), "lr": scheduler.get_last_lr()[0], 
                            "test_top1_acc": top1_test.item(), "test_top5_acc": top5_test.item()})
                # print
                print(f"[Epoch {epoch}/{args.epochs}]\t [Train loss {loss:5f}] [Train Acc@1|5 {top1_train.item():2f}|{top5_train.item():2f}] [Test Acc@1|5 {top1_test.item():2f}|{top5_test.item():2f}]")
                # save model
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')
        
        # dump args
        with open(args.results_dir + '/' + args.wandb_name + ".json", 'w') as fp:
            args = vars(args)
            try:
                del args["device"]
            except:
                pass
            json.dump(args, indent=4, fp=fp)

def train(net, args, epoch, train_loader, criterion, optimizer, scheduler):
    """
    supervised training (to delete)
    """
    net.train()
    scaler = GradScaler(enabled=args.fp16_precision)
    n_iter, train_bar = 0, tqdm(train_loader)
    for images, targets, _, _ in train_bar:
        images = images.to(args.device)
        targets = targets.to(args.device)
        features, mlp_features, logits = net(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 

        if n_iter % args.log_every_n_steps == 0:
            top1_train, top5_train = accuracy(logits, targets, topk=(1, 5))
            train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}, Acc1: {:.4f}'.format(
                                        epoch, 
                                        args.epochs, 
                                        scheduler.get_last_lr()[0],
                                        loss.item(), 
                                        top1_train.item()
                                    ))
        n_iter += 1
    
    # warmup for the first 10 epochs
    if epoch >= 10:
        scheduler.step() 

    return loss, top1_train, top5_train

def test(net, args, test_loader):
    """
    supervised testing (to delete)
    """
    net.eval()
    total_num, top1_acc, top5_acc = 0.0, 0.0, 0.0
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for counter, [imgs, targs, _, _] in enumerate(test_bar):
            imgs = imgs.to(args.device)
            targs = targs.to(args.device)
            _, _, logits = net(imgs)
            top1, top5 = accuracy(logits, targs, topk=(1,5))
            total_num += targs.shape[0]
            top1_acc += top1[0]
            top5_acc += top5[0]
            test_bar.set_description('Testing: ')
        top1_acc /= (counter + 1)
        top5_acc /= (counter + 1)
        return top1_acc, top5_acc 
    
if __name__ == '__main__':
    main()