import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18
from modules.hsic import * 
from functools import partial 
    
class SSL_HSIC(nn.Module):
    """SSL HSIC wrapper """
    def __init__(self, args, dim=128, arch='resnet18', bn_splits=8):
        super(SSL_HSIC, self).__init__()
        self.args = args
        self.net = Model(feature_dim=dim, arch=arch, bn_splits=bn_splits)

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
    def __init__(self, feature_dim=128, arch=None, bn_splits=8):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        # self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
        #                        nn.ReLU(), nn.Linear(512, feature_dim, bias=True))
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(), nn.Linear(512, feature_dim, bias=True))

        # use split batchnorm
        # norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        # resnet_arch = getattr(resnet, arch)
        # net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        # self.net = []
        # for name, module in net.named_children():
        #     if name == 'conv1':
        #         module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #     if isinstance(module, nn.MaxPool2d):
        #         continue
        #     if isinstance(module, nn.Linear):
        #         self.net.append(nn.Flatten(1))
        #     self.net.append(module)

        # self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)
                