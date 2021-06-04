#%% imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import numpy as np
from utils import *
from torch.nn.utils import parameters_to_vector as ptv
from vqgan import VQAutoEncoder, ResBlock, VectorQuantizer, Encoder, Generator
from tqdm import tqdm


#%% helper functions
def approx_difference_function_multi_dim(x, model):
    # eq 4 in paper
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    gx_cur = (gx * x).sum(-1)[:, :, None]
    return gx - gx_cur


# old function - not currently used but keeping for testing purposes
# def get_latents(ae, dataloader):
#     latents = []
#     for x, _ in tqdm(dataloader):
#         x = x.cuda()
#         z = ae.encoder(x) # B, emb_dim, H, W

#         z = z.permute(0, 2, 3, 1).contiguous() # B, H, W, emb_dim
#         z_flattened = z.view(-1, H.emb_dim) # B*H*W, emb_dim

#         # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
#         d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (ae.quantize.embedding.weight**2).sum(1) - \
#             2 * torch.matmul(z_flattened, ae.quantize.embedding.weight.t())
#         min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
#         # print(min_encoding_indices.shape)
#         min_encodings = torch.zeros(min_encoding_indices.shape[0], ae.quantize.codebook_size).to(z)
#         min_encodings.scatter_(1, min_encoding_indices, 1)

#         one_hot = min_encodings.view(z.size(0), H.latent_shape[1], H.latent_shape[2], H.codebook_size)

#         latents.append(one_hot.reshape(one_hot.size(0), -1, H.codebook_size).cpu().contiguous())
    
#     return torch.cat(latents, dim=0)


def generate_latent_ids(ae, dataloader, H):
    latent_ids = []
    for x, _ in tqdm(dataloader):
        x = x.cuda()
        z = ae.encoder(x) # B, emb_dim, H, W

        z = z.permute(0, 2, 3, 1).contiguous() # B, H, W, emb_dim
        z_flattened = z.view(-1, H.emb_dim) # B*H*W, emb_dim

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (ae.quantize.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, ae.quantize.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)

        # this all works ^

        latent_ids.append(min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous())

    latent_ids_out = torch.cat(latent_ids, dim=0)
    print(f'IDs out: {latent_ids_out.shape}')

    return latent_ids_out


def latent_ids_to_onehot(latent_ids, H):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(min_encoding_indices.shape[0], H.codebook_size).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(latent_ids.shape[0], H.latent_shape[1], H.latent_shape[2], H.codebook_size)

    return one_hot.reshape(one_hot.shape[0], -1, H.codebook_size)

#%% define EBM classes
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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.nonlin = lambda x: x * torch.sigmoid(x) # swish
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.downsample = downsample
        if self.downsample:
            self.down_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            
    def forward(self, x):
        h = self.nonlin(self.conv1(x))
        h = x + self.conv2(h)

        if self.downsample:
            h = self.down_conv(self.nonlin(h))
            h = F.avg_pool2d(h, kernel_size=3, stride=2, padding=1)

        return self.nonlin(h)


class ResNetEBM_cat(nn.Module):
    def __init__(self, e_dim, block_str):
        super().__init__()

        blocks = []
        for block_id in block_str:
            if block_id == 'r':
                blocks.append(ResBlock(e_dim, e_dim, downsample=False))
            elif block_id == 'd':
                blocks.append(ResBlock(e_dim, e_dim, downsample=True))
        
        self.net = nn.Sequential(*blocks)
        self.energy_linear = nn.Linear(e_dim, 1)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), out.size(1), -1).mean(-1)
        return self.energy_linear(out).squeeze()


class MyOneHotCategorical:
    def __init__(self, mean):
        self.mean = mean
        self.dist = torch.distributions.OneHotCategorical(probs=self.mean)

    def sample(self, x):
        return self.dist.sample(x)

    def log_prob(self, x):
        logits = self.dist.logits
        lp = torch.log_softmax(logits, -1)
        return (x * lp[None]).sum(-1)


class EBM(nn.Module):
    def __init__(self, net, embedding, codebook_size, emb_dim, latent_shape, mean=None):
        super().__init__()
        self.net = net
        self.embedding = embedding
        self.embedding.requires_grad = False
        self.mean = None if mean is None else nn.Parameter(mean, requires_grad=False)
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.latent_shape = latent_shape

    def embed(self, z):
        z_flattened = z.view(-1, self.codebook_size) # B*H*W, codebook_size
        return torch.matmul(z_flattened, self.embedding.weight).view(
            z.size(0), 
            self.latent_shape[1],
            self.latent_shape[2],
            self.emb_dim
        ).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        # x: B, H*W, codebook_size
        if self.mean is None:
            bd = 0.
        else:
            base_dist = MyOneHotCategorical(self.mean)
            bd = base_dist.log_prob(x).sum(-1)
        
        x = self.embed(x)
        logp = self.net(x).squeeze()
        return logp + bd