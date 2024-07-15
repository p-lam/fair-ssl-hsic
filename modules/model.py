import torch
import torch.nn.functional as F
import os 
import wandb
from tqdm import tqdm
from torch import nn
from torchvision.models import resnet18
from modules.hsic import * 
from functools import partial 
from torchvision.models import resnet18
from torch.cuda.amp import GradScaler, autocast
import time
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from utils import accuracy, save_checkpoint

class SSL_HSIC(nn.Module):
    """SSL HSIC wrapper """
    def __init__(self, *args, **kwargs):
        super(SSL_HSIC, self).__init__()
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def l2_norm(self, x):
        return torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12, out=None)

    def approximate_hsic_zy(self, hidden, h_target, k_type_y='linear'):
        return hsic_regular(hidden, h_target, k_type_y=k_type_y)

    def approximate_hsic_zz(self, z1, z2):
        return hsic_regular(z1, z2)

    def hsic_objective(self, z1, z2, idx, N, sens_att=None):
        target = F.one_hot(idx, num_classes=N)
        hsic_zy = self.approximate_hsic_zy(z1, target, k_type_y='linear')
        hsic_zz = self.approximate_hsic_zz(z1, z2)
        return -hsic_zy + self.args.gamma*torch.sqrt(hsic_zz) 

    def train(self, train_loader, test_loader, N):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        model_checkpoints_folder = self.args.results_dir
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
        n_iter, train_bar = 0, tqdm(train_loader)

        for epoch_counter in range(self.args.epochs):
            for images, targets, sens_att, idx in train_bar:
                im1, im2 = images[0], images[1] 
                im1, im2 = im1.to(self.args.device), im2.to(self.args.device)
                targets = targets.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    z1, mlp_feats1, logits1 = self.model(im1)
                    z2, mlp_feats2, logits2 = self.model(im2) 
                    feat_1, proj_1 = self.l2_norm(z1), self.l2_norm(logits1)
                    feat_2, proj_2 = self.l2_norm(z2), self.l2_norm(logits2)
                    # t1 = time.time()
                    loss = self.hsic_objective(feat_1, feat_2, idx, N, sens_att=sens_att)
                    if torch.isnan(loss):
                        wandb.alert(f'Loss is NaN')     # Will alert you via email or slack that your metric has reached NaN
                        raise Exception(f'Loss is NaN') # This could be exchanged for exit(1) if you do not want a traceback
                    # print(f"Objective took {time.time() - t1} to evaluate")

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update() 
            
                if n_iter % self.args.log_every_n_steps == 0:
                    top1_train, top5_train = accuracy(logits1, targets, topk=(1, 5))
                    train_bar.set_description(f'[Training Epoch {epoch_counter}]')
                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            top1_test, top5_test = self.evaluate(train_loader, test_loader)
        
            # log and print results
            wandb.log({"train_top1_acc": top1_train.item(), "train_top5_acc":top5_train.item(), 
                       "train_loss": loss.item(), "lr": self.scheduler.get_last_lr()[0], 
                       "test_top1_acc": top1_test.item(), "test_top5_acc": top5_test.item()})
            print(f"[Epoch {epoch_counter}/{self.args.epochs}]\t [Train loss {loss:5f}] [Train Acc@1|5 {top1_train.item():2f}|{top5_train.item():2f}] [Test Acc@1|5 {top1_test.item():2f}|{top5_test.item():2f}]")

            filename = os.path.join(model_checkpoints_folder, self.args.wandb_name) + ".pth.tar"
            # save model checkpoints
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=filename, wandb_name=self.args.wandb_name)

        return loss, [feat_1, feat_2, proj_1, proj_2]

    def fit_linear_classifier(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        
        print('Linear Classifier Training...')
        for (name, param) in self.model.named_modules():
            if "fc" not in name:
                param.eval()
        self.model.fc.train()
        self.model.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.model.fc.bias.data.zero_()
        FINETUNE_EPOCHS = 5  # might do slightly better with more training and less LR (e.g. 90 epochs and 0.05 LR in original paper)
        
        opt_fc = torch.optim.SGD(self.model.fc.parameters(), lr=0.3, weight_decay=1e-6, momentum=0.9)  
        schedule_fc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fc, T_max=FINETUNE_EPOCHS)  

        FAST = True  # skip standard training (SGD with data augmentation) and just fit with sklearn
        if FAST:
            X = []
            y = []
            for images, targets, _, _ in train_loader:
                images = images[0]
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    feats, _, _ = self.model(images)
                    X.append(feats.detach().cpu().numpy())
                    y.append(targets.numpy())
            X = np.concatenate(X)
            y = np.concatenate(y)

            fc = LogisticRegression(penalty=None).fit(X, y)
            self.model.fc.weight.data = torch.Tensor(fc.coef_).cuda()
            self.model.fc.bias.data = torch.Tensor(fc.intercept_).cuda()

        else:
            for epoch in tqdm(range(FINETUNE_EPOCHS)):
                for images, targets, _, _ in train_loader:
                    images = images[0]
                    images = images.to(self.args.device)
                    targets = targets.to(self.args.device)

                    with autocast(enabled=self.args.fp16_precision):
                        _, _, pred = self.model(images)
                        loss_sup = self.criterion(pred, targets)

                    opt_fc.zero_grad()
                    scaler.scale(loss_sup).backward()
                    scaler.step(opt_fc)
                    scaler.update()
                schedule_fc.step()

    def evaluate(self, train_loader, test_loader):
        """
        test using a linear classifier on top of backbone features
        """
        self.model.eval()
        total_num, top1_accuracy, top5_accuracy = 0.0, 0.0, 0.0
        # self.fit_linear_classifier(train_loader)
        test_bar = tqdm(test_loader)

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
            
class Fair_SSL_HSIC(SSL_HSIC):
    """Fair SSL HSIC wrapper TODO"""
    def __init__(self, *args, **kwargs):
        super(Fair_SSL_HSIC, self).__init__(*args, **kwargs)

    def approximate_hsic_za(self, hidden, sens_att):
        return hsic_regular(hidden, sens_att)
    
    def hsic_objective(self, z1, z2, idx, N, sens_att):
        target = F.one_hot(idx, num_classes=N)
        hsic_zy = self.approximate_hsic_zy(z1, target, k_type_y='linear')
        hsic_zz = self.approximate_hsic_zz(z1, z2)
        hsic_za = self.approximate_hsic_za(z1, sens_att)
        return -hsic_zy + self.args.gamma*torch.sqrt(hsic_zz) - self.args.lamb*hsic_za
    
class Model(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, num_classes=10, feature_dim=128, arch=None, bn_splits=8):
        super(Model, self).__init__()
        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # pool
        self.pool = nn.AdaptiveAvgPool2d(1)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, feature_dim, bias=False), nn.ReLU())
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.f(x)
        x = self.pool(x).squeeze()
        out = self.g(x)
        logits = self.fc(x.detach())
        return x, F.normalize(out, dim=-1), logits
    
