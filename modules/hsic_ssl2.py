import torch
import torch.nn.functional as F
import numpy as np
import mpmath

# def hsic_inter(x_i, x_j, y_i, y_j):
#     # equation 3 - HSIC(X, Y)
#     N = x_i.shape[0]
#     K = x_i @ x_j.T
#     L = y_i @ y_j.T
#     scale = 1/(N - 1)**2
#     H = torch.eye(N) - 1/N * torch.ones(size=(N, N))
#     hsic = scale * torch.trace(K @ H @ L @ H)
#     return hsic

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)

def hsic_intra(x, y, s_x=1, s_y=1):
    # equation 3 - HSIC(X, Y)
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.double().cuda()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC

def hsic_inter(z_i, z_j, l_i, l_j):
    # \delta{l}/N (1/BM(M-1) \sum{ipl} k(z_i^p, z_i^l) - 1/B^2M^2 \sum_{ijpl} k(z_i^p, z_j^l) - 1/M-1)
    pass

def ssl_hsic_loss(hiddens, kernel_param, num_rff_features, gamma):
    """Compute SSL-HSIC loss."""
    hsic_yz = compute_hsic_yz(hiddens, num_rff_features, kernel_param)
    hsic_zz = compute_hsic_zz(hiddens, num_rff_features, kernel_param)
    return - hsic_yz + gamma * torch.sqrt(hsic_zz)

def compute_hsic_yz(hiddens, num_rff_features, kernel_param):
    """Compute RFF approximation of HSIC_YZ."""
    B = hiddens[0].shape[0]
    M = len(hiddens)
    rff_hiddens = torch.zeros((B, num_rff_features), device=hiddens[0].device)
    mean = torch.zeros((1, num_rff_features), device=hiddens[0].device)
    
    for hidden in hiddens:
        rff_features = imq_rff_features(hidden, num_rff_features, kernel_param)
        rff_hiddens += rff_features
        mean += rff_features.sum(0, keepdim=True)
    
    return (rff_hiddens ** 2).sum() / (B * M * (M - 1)) - (mean ** 2).sum() / (B * M) ** 2

def compute_hsic_zz(hiddens, num_rff_features, kernel_param):
    """Compute RFF approximation of HSIC_ZZ."""
    B = hiddens[0].shape[0]
    M = len(hiddens)
    device = hiddens[0].device
    
    z1_rffs = []
    z2_rffs = []
    center_z1 = torch.zeros((1, num_rff_features), device=device)
    center_z2 = torch.zeros((1, num_rff_features), device=device)
    
    for hidden in hiddens:
        z1_rff = imq_rff_features(hidden, num_rff_features, kernel_param)
        z2_rff = imq_rff_features(hidden, num_rff_features, kernel_param)
        z1_rffs.append(z1_rff)
        center_z1 += z1_rff.mean(0, keepdim=True)
        z2_rffs.append(z2_rff)
        center_z2 += z2_rff.mean(0, keepdim=True)
    
    center_z1 /= M
    center_z2 /= M
    z = torch.zeros((num_rff_features, num_rff_features), device=device, dtype=torch.float32)
    
    for z1_rff, z2_rff in zip(z1_rffs, z2_rffs):
        z += torch.einsum('ni,nj->ij', z1_rff - center_z1, z2_rff - center_z2)
    
    return (z ** 2).sum() / (B * M - 1) ** 2

def imq_rff_features(hidden, num_rff_features, kernel_param):
    """Random Fourier features of IMQ kernel."""
    d = hidden.shape[-1]
    device = hidden.device
    amp, amp_probs = amplitude_frequency_and_probs(d)
    
    amplitudes = torch.from_numpy(np.random.choice(amp, size=(num_rff_features, 1), p=amp_probs)).to(device)
    directions = torch.randn((num_rff_features, d), device=device)
    b = torch.rand((1, num_rff_features), device=device) * 2 * np.pi
    w = directions / directions.norm(dim=-1, keepdim=True) * amplitudes
    z = torch.sqrt(2 / num_rff_features) * torch.cos(hidden / kernel_param @ w.t() + b)
    
    return z

def amplitude_frequency_and_probs(d):
    """Returns frequencies and probabilities of amplitude for RFF of IMQ kernel."""
    if d >= 4096:
        upper = 200
    elif d >= 2048:
        upper = 150
    elif d >= 1024:
        upper = 120
    else:
        upper = 100
    
    x = np.linspace(1e-12, upper, 10000)
    p = compute_prob(d, x)
    return x, p

def compute_prob(d, x_range):
    """Returns probabilities associated with the frequencies."""
    prob = [mpmath.besselk((d - 1) / 2, x) * mpmath.power(x, (d - 1) / 2) for x in x_range]
    normalizer = sum(prob)
    normalized_prob = [float(x / normalizer) for x in prob]
    return np.array(normalized_prob)