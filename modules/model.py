import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18
from modules.hsic import * 
from functools import partial 
from torchvision.models import resnet18

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

    def hsic_objective(self, z1, z2, idx, N):
        target = F.one_hot(idx, num_classes=N)
        hsic_zy = self.approximate_hsic_zy(z1, target, k_type_y='linear')
        hsic_zz = self.approximate_hsic_zz(z1, z2)
        return -hsic_zy + self.args.gamma*torch.sqrt(hsic_zz) 

    def forward(self, im1, im2, idx, N):
        z1, pred1 = self.net(im1)
        z2, pred2 = self.net(im2) 
        feat_1, proj_1 = self.l2_norm(z1), self.l2_norm(pred1)
        feat_2, proj_2 = self.l2_norm(z2), self.l2_norm(pred2)
        loss = self.hsic_objective(feat_1, feat_2, idx, N)
        return loss, [feat_1, feat_2, proj_1, proj_2]

class Fair_SSL_HSIC(SSL_HSIC):
    """Fair SSL HSIC wrapper TODO"""
    def __init__(self):
        super(Fair_SSL_HSIC, self).__init__()

    def approximate_hsic_za(self, hidden, sens_att):
        return hsic_normalized_cca(hidden, sens_att)

    def hsic_objective(self, z1, z2, target, sens_att):
        hsic_zy = self.approximate_hsic_zy(self, z1, target)
        hsic_zz = self.approximate_hsic_zz(self, z1, z2)
        hsic_za = self.approximate_hsic_za(self, z1, sens_att)
        return -hsic_zy + self.gamma*torch.sqrt(hsic_zz) - self.lamb*hsic_za

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
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.ReLU())
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.f(x)
        x = self.pool(x).squeeze()
        out = self.g(x)
        logits = self.fc(x)
        return F.normalize(x, dim=-1), F.normalize(out, dim=-1), logits
    
