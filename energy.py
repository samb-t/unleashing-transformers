import torch
import torch.nn as nn
import torch.distributions as dists
import torchvision
import numpy as np
from vqgan import VQAutoEncoder
from tqdm import tqdm
import os

def approx_difference_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    wx = gx * -(2. * x - 1)
    return wx.detach()

# Gibbs-With-Gradients for binary data
class DiffSampler(nn.Module):
    def __init__(self, dim, n_steps=1, temp=2., step_size=1.0):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.temp = temp
        self.step_size = step_size
        self.diff_fn = lambda x, m: approx_difference_function(x, m) / self.temp


    def step(self, x, model, mask):
        x_cur = x
        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model) - 1e9 * mask
            cd_forward = dists.OneHotCategorical(logits=forward_delta)
            changes = cd_forward.sample()

            lp_forward = cd_forward.log_prob(changes)

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

            reverse_delta = self.diff_fn(x_delta, model) - 1e9 * mask
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

        return x_cur

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.nonlin = lambda x: x * torch.sigmoid(x)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

    def forward(self, x):
        out = self.nonlin(self.conv1(x))
        out = x + self.conv2(out)
        return self.nonlin(out)

class ResNetEBM_cat(nn.Module):
    def __init__(self, n_channels=64):
        super().__init__()
        self.net = nn.Sequential(*[BasicBlock(n_channels, n_channels, 1) for _ in range(6)])
        self.energy_linear = nn.Linear(n_channels, 1)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), out.size(1), -1).mean(-1)
        return self.energy_linear(out).squeeze()

class MyOneHotCategorical:
    def __init__(self, mean):
        self.dist = torch.distributions.OneHotCategorical(probs=mean)

    def sample(self, x):
        return self.dist.sample(x)

    def log_prob(self, x):
        logits = self.dist.logits
        lp = torch.log_softmax(logits, -1)
        return (x * lp[None]).sum(-1)

class EBM(nn.Module):
    def __init__(self, net, embedding, mean=None):
        super().__init__()
        self.net = net
        self.embedding = embedding
        self.embedding.requires_grad = False
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)
    
    def embed(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, 64)
        return torch.matmul(z_flattened, self.embedding.weight).view(z.size(0), 8, 8, 64).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            base_dist = MyOneHotCategorical(self.mean)
            bd = base_dist.log_prob(x).sum(-1)
        
        x = self.embed(x)
        logp = self.net(x).squeeze()
        return logp + bd


def get_latents(ae, dataloader):
    latents = []
    for x, _ in tqdm(dataloader):
        x = x.cuda()
        z = ae.encoder(x)

        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, 64)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (ae.quantize.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, ae.quantize.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], ae.quantize.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        one_hot = min_encodings.view(z.size(0), 8, 8, 10).permute(0, 3, 1, 2).contiguous().cpu()

        latents.append(one_hot)
    
    return torch.cat(latents, dim=0)


input_size = [1, 8, 8]
data_dim = np.prod(input_size)
sampler = DiffSampler(data_dim, 1)

ae = VQAutoEncoder(1, 64, 10)
ae = torch.load('autoencoder.pkl').cuda()

net = ResNetEBM_cat()
energy = EBM(net, ae.quantize.embedding.weight.data).cuda() # 10x64
optim = torch.optim.Adam(energy.parameters())

transform = torchvision.transforms.Compose([torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.MNIST('~/workspace/data', train=True, transform=transform, download=True)

if os.path.exists('latents.pkl'):
    latents = torch.load('latents.pkl')
else:
    full_dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)
    latents = get_latents(ae, full_dataloader)
    torch.save(latents, 'latents.pkl')


intit_batch # B, 256, H*W
eps = 1e-2 / 256
init_mean = init_batch.mean(0) + eps # 256, H*W
init_mean = init_mean / init_mean.sum(-1)[:, None]


dataloader = torch.utils.data.DataLoader(latents, batch_size=128, shuffle=True)

global_step = 0
for epoch in range(10000):
    for x, _ in dataloader:

        

        global_step += 1
