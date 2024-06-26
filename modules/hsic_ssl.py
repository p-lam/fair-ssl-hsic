"""Helpers for computing SSL-HSIC losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mpmath
import sklearn

from numpy.random import rand
from scipy.linalg import hadamard
from time import time

def pairwise_distance_square(x: torch.Tensor, y: torch.Tensor, maximum=1e10) -> torch.Tensor:
    """Computes the square of pairwise distances."""
    x_sq = torch.einsum('ij,ij->i', x, x)[:, None]
    y_sq = torch.einsum('ij,ij->i', y, y)[None, :]
    x_y = torch.einsum('ik,jk->ij', x, y)
    dist = x_sq + y_sq - 2 * x_y
    return torch.clamp(dist, 0.0, maximum)

def get_label_weights(batch: int):
    """Returns the positive and negative weights of the label kernel matrix."""
    w_pos_base = torch.tensor(1.0).view(1, 1)
    w_neg_base = torch.tensor(0.0).view(1, 1)
    w_mean = (w_pos_base + w_neg_base * (batch - 1)) / batch
    w_pos_base -= w_mean
    w_neg_base -= w_mean
    w_mean = (w_pos_base + w_neg_base * (batch - 1)) / batch
    w_pos = w_pos_base - w_mean
    w_neg = w_neg_base - w_mean
    return w_pos.item(), w_neg.item()

def compute_prob(n: int, x_range: np.ndarray) -> np.ndarray:
    """Compute the probability to sample the random fourier features."""
    probs = [mpmath.besselk((n - 1) / 2, x) * mpmath.power(x, (n - 1) / 2) for x in x_range]
    normalized_probs = [float(p / sum(probs)) for p in probs]
    return np.array(normalized_probs)

def imq_amplitude_frequency_and_probs(n: int):
    """Returns the range and probability for sampling RFF."""
    x = np.linspace(1e-12, 100, 10000)
    p = compute_prob(n, x)
    return x, p

def imq_rff_features(num_features: int, rng: torch.Generator, x: torch.Tensor, c: float) -> torch.Tensor:
    """Returns the RFF feature for IMQ kernel with pre-computed amplitude prob."""
    d = x.shape[-1]
    amp, amp_probs = imq_amplitude_frequency_and_probs(d)
    amp = torch.tensor(np.random.choice(amp, size=(num_features, 1), p=amp_probs), device=x.device, dtype=x.dtype)
    directions = torch.randn((num_features, d), generator=rng, device=x.device, dtype=x.dtype)
    b = torch.rand((1, num_features), generator=rng, device=x.device) * 2 * np.pi
    w = directions / directions.norm(dim=-1, keepdim=True) * amp
    z_x = np.sqrt(2 / num_features) * torch.cos(x / c @ w.t() + b)
    return z_x

def rff_approximate_hsic_xy(list_hiddens, w, num_features, rng, c, rff_kwargs, kernel='gaussian'):
    """RFF approximation of Unbiased HSIC(X, Y)."""
    b, _ = list_hiddens[0].shape
    k = len(list_hiddens)
    rff_hiddens = torch.zeros((b, num_features), device=list_hiddens[0].device)
    mean = torch.zeros((1, num_features), device=list_hiddens[0].device)
    n_square = (b * k) ** 2
    for hidden in list_hiddens:
        if kernel == 'gaussian':
            pass
        else:
            rff_features = imq_rff_features(num_features, rng, hidden, c, **rff_kwargs)
        rff_hiddens += rff_features
        mean += rff_features.sum(0, keepdim=True)
    return w * ((rff_hiddens**2).sum() / (b * k * (k - 1)) - (mean**2).sum() / n_square)

def rff_approximate_hsic_xx(list_hiddens, num_features, rng, rng_used, c, rff_kwargs, kernel='gaussian'):
    """RFF approximation of HSIC(X, X) where inverse multiquadric kernel is used."""
    x1_rffs = []
    x2_rffs = []

    for xs in list_hiddens:
        if kernel == 'gaussian':
            sigma = np.median(sklearn.metrics.pairwise.pairwise_distances(xs.detach().numpy()))
            x1_rff = random_feature(sigma=sigma, random_feature_method='rff')
        else:
            x1_rff = imq_rff_features(num_features, rng_used, xs, c, **rff_kwargs)
            x1_rffs.append(x1_rff)
            x2_rff = imq_rff_features(num_features, rng, xs, c, **rff_kwargs)
            x2_rffs.append(x2_rff)

    mean_x1 = torch.mean(torch.stack(x1_rffs), dim=0, keepdim=True).mean(0, keepdim=True)
    mean_x2 = torch.mean(torch.stack(x2_rffs), dim=0, keepdim=True).mean(0, keepdim=True)
    z = torch.zeros((num_features, num_features), device=list_hiddens[0].device, dtype=torch.float32)
    for x1_rff, x2_rff in zip(x1_rffs, x2_rffs):
        z += torch.einsum('ni,nj->ij', x1_rff - mean_x1, x2_rff - mean_x2)
    return (z ** 2).sum() / ((x1_rff.shape[0] * len(list_hiddens)) ** 2)

class random_feature:
    """Referenced from: https://github.com/endsley/ml_examples/blob/master/Random_Features/random_features.ipynb"""
    # sample_num, the larger the better approximation
    # random_feature_method can be 'orthogonal' (default) or 'rff'
    def __init__(self, kernel='rbk', sigma=1, random_feature_method='orthogonal', sample_num=482):
        self.method = random_feature_method
        self.kernel = kernel
        self.m = sample_num		# the number of samples used to approximate k(x,y)
        self.θ = None			# phase shift
        self.σ = sigma
    #
    def sample_rademacher_distribution(self, num_of_samples):
        r = np.random.binomial(1, 0.5, int(num_of_samples))
        r = 2*(r - 0.5)
        D = np.diag(r)
        return D
    #
    def initialize_random_features(self, X):
        if self.θ == None:
            X0 = X
            N = X.shape[0]
            d = X.shape[1]
            σˉᑊ = 1/self.σ
            ƻπ = 2*np.pi
            m = self.m
            dᑊᐟᒾ = np.sqrt(d)

            if self.method == 'orthogonal':	# perform SORF
                #	Here we will pad the data to ensure data dimension is in power of 2 for Hadamard matrix
                ḿ = np.power(2, np.ceil(np.log2(d))) 	# dimension in power of 2 just greater than d
                Δ = int(ḿ - d)
                if Δ != 0:
                    pad0 = np.zeros((N, Δ))
                    X0 = np.concatenate((X, pad0), axis=1)
                #	each loop creates ḿ out of m, we need to Figure out how many times we need to regenerate 
                #	Note the sample_num may not be equal to the actual random feature width M
                #		m = the number of random features width suggested by the user
                #		ḿ = the number of random features generated by each hadamard matrix 
                #		M = the acutal number of random features we actually use since power of 2 is required by hadamard
                repeat = int(np.ceil(m/ḿ))
                W = np.empty((int(ḿ),0))
                M = int(repeat*ḿ)
                for ɷ in range(repeat):
                    #	Generate H and diagonal matrix D sampled from rademacher distribution
                    H = 1/np.sqrt(ḿ)*hadamard(ḿ)		# normalized H to be orthonormal
                    Dᑊ = self.sample_rademacher_distribution(ḿ)
                    Dᒾ = self.sample_rademacher_distribution(ḿ)
                    Dᶾ = self.sample_rademacher_distribution(ḿ)
                    #
                    HDᑊHDᒾHDᶾ = H.dot(Dᑊ).dot(H).dot(Dᒾ).dot(H).dot(Dᶾ)
                    Ŵ = (dᑊᐟᒾ*σˉᑊ*HDᑊHDᒾHDᶾ).T			# we did follow the paper of Wx and use XW format
                    W = np.concatenate((W, Ŵ), axis=1)
                    #
                θ = ƻπ*rand(1, M)
                return [W, X0, M, θ]
                #
            elif self.method == 'rff':
                θ = ƻπ*rand(1, m)
                W = (σˉᑊ)*np.random.randn(d, m)	# random projection matrix W
                return [W, X, m, θ]

    def get_feature_map(self, X):
        [W,Ẋ, m, θ] = self.initialize_random_features(X)
        #
        ẊW = Ẋ.dot(W)
        c = np.sqrt(2.0/m)	
        return c*np.cos(ẊW + θ)	

    def get_kernel(self, X):
        Φ = self.get_feature_map(X)
        K = Φ.dot(Φ.T)
        K = np.maximum(0,K)	# run a relu on the kernel so no negative values
        if self.kernel == 'rbk': K = np.minimum(1,K) # make sure the kernel values doesn't go beyond 1 for gaussian
        return K

class HSICLoss2(nn.Module):
    """SSL-HSIC loss."""
    
    def __init__(self, num_rff_features: int, regul_weight: float, name = 'hsic_loss'):
        """Initialize HSICLoss."""
        super().__init__()
        self.num_rff_features = num_rff_features
        self.regul_weight = regul_weight
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, list_hiddens, rff_kwargs = None):
        """Returns the HSIC loss and summaries."""
        start_time = time()
        b = list_hiddens[0].shape[0]
        c = self.scale.item()
        rff_kwargs = rff_kwargs or {}
        w_pos, w_neg = get_label_weights(b)

        # rng1 = torch.Generator(device=list_hiddens[0].device)
        # rng2 = torch.Generator(device=list_hiddens[0].device)
        # rng1.manual_seed(torch.randint(0, 10000, (1,)).item())
        # rng2.manual_seed(torch.randint(0, 10000, (1,)).item())

        hsic_xy = rff_approximate_hsic_xy(list_hiddens, w_pos - w_neg, self.num_rff_features, rng1, c, rff_kwargs=rff_kwargs)
        print(f"HSIC_XY computed after {time() - start_time} seconds")
        hsic_xx = rff_approximate_hsic_xx(list_hiddens, self.num_rff_features, rng1, rng2, c, rff_kwargs)
        print(f"HSIC_XX computed after {time() - start_time} seconds")
        total_loss = self.regul_weight * torch.sqrt(hsic_xx) - hsic_xy

        # Compute gradient norm.
        n_samples = int(1024 / len(list_hiddens))
        sampled_hiddens_1 = torch.cat([x[torch.randint(b, (n_samples,), generator=rng1)] for x in list_hiddens])
        sampled_hiddens_2 = torch.cat([x[torch.randint(b, (n_samples,), generator=rng2)] for x in list_hiddens])
        dist_sq = pairwise_distance_square(sampled_hiddens_1, sampled_hiddens_2)
        print(f"Pairwise distances computed after {time() - start_time} seconds")
        grad = torch.autograd.grad((dist_sq / torch.sqrt(self.scale + dist_sq**2)).sum(), self.scale)[0]
        grad_norm = 0.5 * torch.log(torch.clamp(grad ** 2, min=1e-14)).mean()
        summaries = {
            'kernel_loss/hsic_xy': hsic_xy,
            'kernel_loss/hsic_xx': hsic_xx,
            'kernel_loss/total_loss': total_loss,
            'kernel_loss/kernel_param': self.scale,
            'kernel_loss/grad_norm': grad_norm
        }
        print(f"Final loss computed after {time() - start_time} seconds")
        return total_loss, summaries

class HSICLoss(nn.Module):
    """SSL-HSIC loss."""
    
    def __init__(self, num_rff_features: int, regul_weight: float, name = 'hsic_loss'):
        """Initialize HSICLoss."""
        super().__init__()
        self.num_rff_features = num_rff_features
        self.regul_weight = regul_weight
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, list_hiddens, rff_kwargs = None):
        """Returns the HSIC loss and summaries."""
        start_time = time()
        b = list_hiddens[0].shape[0]
        c = self.scale.item()
        rff_kwargs = rff_kwargs or {}
        w_pos, w_neg = get_label_weights(b)
        rng1 = torch.Generator(device=list_hiddens[0].device)
        rng2 = torch.Generator(device=list_hiddens[0].device)
        rng1.manual_seed(torch.randint(0, 10000, (1,)).item())
        rng2.manual_seed(torch.randint(0, 10000, (1,)).item())
        hsic_xy = rff_approximate_hsic_xy(list_hiddens, w_pos - w_neg, self.num_rff_features, rng1, c, rff_kwargs=rff_kwargs)
        print(f"HSIC_XY computed after {time() - start_time} seconds")
        hsic_xx = rff_approximate_hsic_xx(list_hiddens, self.num_rff_features, rng1, rng2, c, rff_kwargs)
        print(f"HSIC_XX computed after {time() - start_time} seconds")
        total_loss = self.regul_weight * torch.sqrt(hsic_xx) - hsic_xy

        # Compute gradient norm.
        n_samples = int(1024 / len(list_hiddens))
        sampled_hiddens_1 = torch.cat([x[torch.randint(b, (n_samples,), generator=rng1)] for x in list_hiddens])
        sampled_hiddens_2 = torch.cat([x[torch.randint(b, (n_samples,), generator=rng2)] for x in list_hiddens])
        dist_sq = pairwise_distance_square(sampled_hiddens_1, sampled_hiddens_2)
        print(f"Pairwise distances computed after {time() - start_time} seconds")
        grad = torch.autograd.grad((dist_sq / torch.sqrt(self.scale + dist_sq**2)).sum(), self.scale)[0]
        grad_norm = 0.5 * torch.log(torch.clamp(grad ** 2, min=1e-14)).mean()
        summaries = {
            'kernel_loss/hsic_xy': hsic_xy,
            'kernel_loss/hsic_xx': hsic_xx,
            'kernel_loss/total_loss': total_loss,
            'kernel_loss/kernel_param': self.scale,
            'kernel_loss/grad_norm': grad_norm
        }
        print(f"Final loss computed after {time() - start_time} seconds")
        return total_loss, summaries