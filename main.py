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
from dataloader import get_celeba_dataset, ColoredMNIST
from tqdm import tqdm
from utils import *
from modules.model import SSL_HSIC, Fair_SSL_HSIC
from simclr.simclr import SimCLR
from simclr.resnet_simclr import ResNetSimCLR
from sklearn.linear_model import LogisticRegression

# locating dataset folder(s)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def parse_args():
    parser = argparse.ArgumentParser(description='Train Fair SSL-HSIC')
    # dataset arguments
    parser.add_argument('--dataset', dest='dataset', default='cmnist', type=str, help='dataset (options: cmnist, celeba, cifar10)', choices=['cmnist', 'celeb', 'cifar10'])    
    parser.add_argument("--data", help='path for loading data', default='data', type=str)
    # model arguments
    parser.add_argument("--num_workers", help="number of workers", default=4, type=int)
    parser.add_argument('--model', default='ssl-hsic', type=str, help='Model to use', choices=['simclr','ssl-hsic','fair-ssl-hsic'])
    parser.add_argument('-a', '--arch', default='resnet18', help='resnet architecture')
    # training args/hyperparameters
    parser.add_argument('--lr', '--learning-rate', default=3.0, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--schedule', default=[600, 900], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
    parser.add_argument('--feature_dim', default=64, type=int, help='Feature dim for latent vector')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=5, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--start_epoch', default=1, type=int, help='Starting epoch')
    parser.add_argument('--lambda', default=1, type=int, help='Regularization Coefficient')
    parser.add_argument('--gamma', default=3, type=int, help='Regularization Coefficient')
    parser.add_argument('--hsic_type', default='regular', type=str, help='type of hsic approx: regular, normalized, normalized cca') 
    parser.add_argument('--n_views', default=2, type=int, help="number of augmentations for simclr/ssl") 
    parser.add_argument('--temperature', default=0.07, type=float, help="contrastive temperature") 
    # misc arguments
    parser.add_argument('--bn_splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')
    parser.add_argument('--results-dir', default='./results', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')   
    parser.add_argument('--log_every_n_steps', default=1, type=int) 
    parser.add_argument('--fp16_precision', action='store_true') 
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str) 
    parser.add_argument("--wandb_name", default="train_1")
    # parse and return args
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available()==True else 'cpu'
    device = torch.device(device)

    # setup tracking in wandb
    args_dict = deepcopy(vars(args)) 
    print(f"Saving to wandb under run name: {args.wandb_name}")
    
    wandb.init(
        project="SSL-HSIC",
        name=f"{args.wandb_name}" if args.wandb_name is not None else None,
        config=args_dict
    )

    # load and transform data
    if args.dataset == "celeba":
        dataset = get_celeba_dataset(n_views=args.n_views)
        # split dataset into train/val/test
        train_size = ceil(0.70 * len(dataset))
        val_size = ceil(0.10 * len(dataset))
        test_size = len(dataset) - (train_size+val_size)
        train_, test_dataset = torch.utils.data.random_split(dataset, [train_size+val_size, test_size])
        train_dataset, val_dataset = torch.utils.data.random_split(train_, [train_size, val_size])
        # set training/testing augmentations for the datasets
        train_dataset.dataset.set_transform_mode(train=True)
        val_dataset.dataset.set_transform_mode(train=False)
        test_dataset.dataset.set_transform_mode(train=False)
    elif args.dataset == "cmnist": 
        dataset_type = ColoredMNIST # if args.model in ['ssl-hsic', "simclr"] else ColoredMNISTVanilla
        train_dataset = dataset_type(root='data', env='train', n_views=args.n_views)
        # dataset_len = len(dataset)
        # train_size = ceil(0.80 * len(dataset))
        # val_size = ceil(0.20 * len(dataset))
        # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        test_dataset = dataset_type(root='data', env='test') 

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # load model
    if args.model == 'ssl-hsic':
        model = SSL_HSIC(args, dim=args.feature_dim, arch=args.arch, bn_splits=args.bn_splits).to(device)
    elif args.model == 'simclr':
        model = ResNetSimCLR(base_model=args.arch, out_dim = 512)
    else:
        print(f"Model {args.model} not implemented!")

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    
    if args.model == 'simclr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=len(train_loader), eta_min=0,
                                                                last_epoch=-1)
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)
    else:
        # load model if resume
        if args.resume != '':
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print('Loaded from: {}'.format(args.resume))

        # logging
        results = {'train_loss': [], 'test_acc@1': []}
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)
        # dump args
        with open(args.results_dir + '/args.json', 'w') as fid:
            json.dump(args.__dict__, fid, indent=2)
            # training loop
        for epoch in range(args.start_epoch, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer, device, epoch, len(train_dataset), args)
            results['train_loss'].append(train_loss)
            test_acc_1 = test(model, train_loader, test_loader, epoch, device, len(test_dataset), args)
            results['test_acc@1'].append(test_acc_1)
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(args.start_epoch, epoch+1))
            data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
            # save model
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')

def train(net, train_loader, optimizer, device, epoch, dataset_len, args):
    net.train()
    adjust_learning_rate(optimizer, epoch, args)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)

    # train model
    for pos_1, pos_2, label, sen_attr, dataset_idx in train_bar:
        pos_1, pos_2, label = pos_1.to(device), pos_2.to(device), label.to(device)
        # batch_pos = torch.eye(pos_1.shape[0]).to(device)
        loss, list_hiddens = net(pos_1, pos_2, dataset_idx, dataset_len)
        feat_1, feat_2, proj_1, proj_2 = list_hiddens[0], list_hiddens[1], list_hiddens[2], list_hiddens[3]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += train_loader.batch_size
        total_loss += loss.item() * train_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num 

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def fit_linear_classifier(net, train_loader, dataset_len, device):
    net.eval()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)

    # train model
    feats = []
    labels = []
    with torch.no_grad():
        for pos_1, pos_2, label, sen_attr, dataset_idx in train_bar:
            pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
            loss, list_hiddens = net(pos_1, pos_2, dataset_idx, dataset_len)
            feats.append(list_hiddens[1].detach().cpu().numpy())
            labels.append(label.cpu().numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    linear_model = LogisticRegression().fit(feats, labels)
    return linear_model.coef_, linear_model.intercept_

def test(net, train_loader, test_loader, epoch, device, dataset_len, args):
    """
    test using a linear classifier on top of backbone features
    """
    net.eval()
    total_num, test_bar = 0.0, tqdm(test_loader)
    top1_accuracy, top5_accuracy = 0.0, 0.0

    # fit linear classifier for evaluation
    w_cls, b_cls = fit_linear_classifier(net, train_loader, dataset_len, device)
    w_cls = torch.Tensor(w_cls).to(device)
    b_cls = torch.Tensor(b_cls).to(device)
    
    # calculate accuracy
    with torch.no_grad():
        for counter, [x, y, labels, sens_att, dataset_idx] in enumerate(test_bar):
            x,y = x.to(device), y.to(device)
            labels = labels.to(device)
            loss, list_hiddens = net(x, y, dataset_idx, dataset_len)
            feat_1, feat_2, proj_1, proj_2 = list_hiddens[0], list_hiddens[1], list_hiddens[2], list_hiddens[3]
            logits = feat_2 @ w_cls.T + b_cls
            top1, top5 = accuracy(logits, labels, topk=(1,5))
            total_num += test_loader.batch_size
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(epoch, args.epochs, top1_accuracy, top5_accuracy))
        print('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(epoch, args.epochs, top1_accuracy, top5_accuracy))
        return top1_accuracy.item()
    
if __name__ == '__main__':
    main()