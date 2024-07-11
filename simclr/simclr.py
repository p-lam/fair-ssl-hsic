import os
import sys
import wandb
import torch
import torch.nn.functional as F
import numpy as np 

from sklearn.linear_model import LogisticRegression
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import accuracy, save_checkpoint, save_config_file
from modules.model import Model

torch.manual_seed(0)

class SimCLR(object):
    """
    SimCLR wrapper 
    Referenced from: https://github.com/sthalles/SimCLR/tree/master
    """
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        # normalize features
        features = F.normalize(features, dim=-1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        # return logits and labels
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, test_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        model_checkpoints_folder = self.args.results_dir
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
        n_iter, train_bar = 0, tqdm(train_loader)

        for epoch_counter in range(self.args.epochs):
            for images, targets, _, _ in train_bar:
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features, mlp_features, _ = self.model(images)
                    logits, labels = self.info_nce_loss(mlp_features)
                    loss = self.criterion(logits, labels)

                    # train fc layer (using detach to not affect training)
                    # just for monitoring accuracy of representations
                    pred = self.model.fc(features.detach()[:targets.shape[0]])
                    loss_sup = self.criterion(pred, targets)
    
                self.optimizer.zero_grad()
                scaler.scale(loss + loss_sup).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1_train, top5_train = accuracy(pred, targets, topk=(1, 5))
                    train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}, Acc1: {:.4f}'.format(
                                                epoch_counter, 
                                                self.args.epochs, 
                                                self.scheduler.get_last_lr()[0],
                                                loss.item(), 
                                                top1_train.item()
                                            ))
                n_iter += 1
            
            top1_test, top5_test = self.evaluate(train_loader, test_loader)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            # log and print results
            wandb.log({"train_top1_acc": top1_train.item(), "train_top5_acc":top5_train.item(), 
                       "train_loss": loss.item(), "lr": self.scheduler.get_last_lr()[0], 
                       "test_top1_acc": top1_test.item(), "test_top5_acc": top5_test.item()})
            print(f"[Epoch {epoch_counter}/{self.args.epochs}]\t [Train loss {loss:5f}] [Train Acc@1|5 {top1_train.item():2f}|{top5_train.item():2f}] [Test Acc@1|5 {top1_test.item():2f}|{top5_test.item():2f}]")

        # save model checkpoints
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(model_checkpoints_folder, self.args.wandb_name))

    def fit_linear_classifier(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        n_iter, train_bar = 0, tqdm(train_loader)
        train_bar.set_description('Linear Classifier Training...')
        opt_fc = torch.optim.SGD(self.model.fc.parameters(), lr=self.args.lr, weight_decay=self.args.wd, momentum=0.9)    
        for epoch_counter in range(5):
            for images, targets, _, _ in train_bar:
                images = images[0]
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features, _, _ = self.model(images)
                    pred = self.model.fc(features.detach())
                    loss_sup = self.criterion(pred, targets)

                self.optimizer.zero_grad()
                scaler.scale(loss_sup).backward()
                scaler.step(self.optimizer)
                scaler.update()

    def evaluate(self, train_loader, test_loader):
        """
        test using a linear classifier on top of backbone features
        """
        self.model.eval()
        total_num, top1_accuracy, top5_accuracy = 0.0, 0.0, 0.0
        test_bar = tqdm(test_loader)

        # fit linear classifier for evaluation
        self.fit_linear_classifier(train_loader)
        # w_cls = torch.Tensor(w_cls).to(self.args.device)
        # b_cls = torch.Tensor(b_cls).to(self.args.device)
        
        # calculate accuracy
        with torch.no_grad():
            with autocast(enabled=self.args.fp16_precision):
                for counter, [imgs, targs, _, _] in enumerate(test_bar): 
                    imgs = imgs.to(self.args.device)
                    targs = targs.to(self.args.device)

                    _, _, logits = self.model(imgs)
                    loss = self.criterion(logits, targs)

                    top1, top5 = accuracy(logits, targs, topk=(1,5))
                    total_num += targs.shape[0]
                    top1_accuracy += top1[0]
                    top5_accuracy += top5[0]
                    test_bar.set_description('Testing: ')
                top1_accuracy /= (counter + 1)
                top5_accuracy /= (counter + 1)
                return top1_accuracy, top5_accuracy
        
