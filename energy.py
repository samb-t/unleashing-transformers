#%% imports
import torch
import torch.nn as nn
import torch.distributions as dists
import torchvision
import numpy as np
from vqgan import VQAutoEncoder, ResBlock, VectorQuantizer, Encoder, Generator
from tqdm import tqdm
import os
import visdom
from utils import *

#%% hparams
dataset = 'mnist'
if dataset == 'minist':
    batch_size = 128
    buffer_size = 10000
    sampling_steps = 50
    warmup_iters = 2000
    main_lr = 1e-4

vis = visdom.Visdom()

def approx_difference_function_multi_dim(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    gx_cur = (gx * x).sum(-1)[:, :, None]
    return gx - gx_cur

# TODO: Try multiple flips per step
# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, dim, n_steps=1, approx=True, temp=2.):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        self.diff_fn = lambda x, m: approx_difference_function_multi_dim(x, m) / self.temp

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - 1e9 * x_cur
            #print(forward_logits)
            cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
            changes = cd_forward.sample()

            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out cuanged dim and add in the change
            x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - 1e9 * x_delta
            cd_reverse = dists.OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).sum(-1).mean().item()
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
        self.mean = None if mean is None else nn.Parameter(mean, requires_grad=False)
    
    def embed(self, z):
        # z: B, H*W, 10
        # z = z.permute(0, 2, 3, 1).contiguous() # B, H, W, 10
        z_flattened = z.view(-1, 10) # B*H*W, 10
        return torch.matmul(z_flattened, self.embedding.weight).view(z.size(0), 8, 8, 64).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        # x: B, H*W, 10
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
        z = ae.encoder(x) # B, 64, H, W

        z = z.permute(0, 2, 3, 1).contiguous() # B, H, W, 64
        z_flattened = z.view(-1, 64) # B*H*W, 64

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (ae.quantize.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, ae.quantize.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], ae.quantize.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        one_hot = min_encodings.view(z.size(0), 8, 8, 10)

        latents.append(one_hot.reshape(one_hot.size(0), -1, 10).cpu().contiguous())
    
    return torch.cat(latents, dim=0)


input_size = [1, 8, 8]
data_dim = np.prod(input_size)
sampler = DiffSamplerMultiDim(data_dim, 1)

ae = VQAutoEncoder(1, 64, 10)
ae = load_model(ae, 'ae', 10000)

if os.path.exists('latents.pkl'):
    latents = torch.load('latents.pkl')
else:
    full_dataloader = get_data_loader('mnist', img_size, batch_size, drop_last=False)
    latents = get_latents(ae, full_dataloader)
    torch.save(latents, 'latents.pkl')

dataloader = torch.utils.data.DataLoader(latents, batch_size=batch_size, shuffle=True)

# latents # B, H*W, 10
eps = 1e-3 / 10
init_mean = latents.mean(0) + eps # H*W, 10
init_mean = init_mean / init_mean.sum(-1)[:, None] # renormalize pdfs after adding eps

init_dist = MyOneHotCategorical(init_mean)
buffer = init_dist.sample((buffer_size,)) # 1000, H*W, 10
all_inds = list(range(buffer_size))

# create energy model
net = ResNetEBM_cat()
energy = EBM(net, ae.quantize.embedding, mean=init_mean).cuda() # 10x64
optim = torch.optim.Adam(energy.parameters())

# check reocnstructions are correct
latent_batch = next(iter(dataloader))[:64].cuda()
quant = energy.embed(latent_batch)
recons = ae.generator(quant)
vis.images(recons.clamp(0,1), win='recon_check', opts=dict(title='recon_check'))

hop_dists = []

global_step = 0
for epoch in range(10000):
    for x in dataloader:
        # linearly anneal in learning rate
        if global_step < warmup_iters:
            lr = main_lr * float(global_step) / warmup_iters
            for param_group in optim.param_groups:
                param_group['lr'] = lr
        
        x = x.cuda()#.requires_grad_()

        buffer_inds = sorted(np.random.choice(all_inds, batch_size, replace=False))
        x_buffer = buffer[buffer_inds].cuda()
        x_fake = x_buffer

        hops = []  # keep track of how much the sampler moves particles around
        for k in range(sampling_steps):
            x_fake_new = sampler.step(x_fake.detach(), energy).detach()
            h = (x_fake_new != x_fake).float().view(x_fake_new.size(0), -1).sum(-1).mean().item()
            hops.append(h)
            x_fake = x_fake_new
        hop_dists.append(np.mean(hops))

        # update buffer
        buffer[buffer_inds] = x_fake.detach().cpu()

        logp_real = energy(x).squeeze()
        logp_fake = energy(x_fake).squeeze()

        obj = logp_real.mean() - logp_fake.mean()

        loss = -obj #+ 0.01 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())

        optim.zero_grad()
        loss.backward()
        optim.step()

        if global_step % 20 == 0:
            print(f"Step: {global_step}, log p(real)={logp_real.mean():.4f}, log p(fake)={logp_fake.mean():.4f}, diff={obj:.4f}, hops={hop_dists[-1]:.4f}")

            q = energy.embed(x_fake)
            samples = ae.generator(q)
            vis.images(samples[:64].clamp(0,1), win='samples', opts=dict(title='samples'))

        global_step += 1
