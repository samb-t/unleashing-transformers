#%% imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import numpy as np
from .sampler import Sampler
from .helpers import MyOneHotCategorical

# TODO: figure out if this can be seperated without messing up imports
def latent_ids_to_onehot(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(
        min_encoding_indices.shape[0],
        codebook_size
    ).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(
        latent_ids.shape[0],
        latent_shape[1],
        latent_shape[2],
        codebook_size
    )
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)


#%% helper functions
def approx_difference_function_multi_dim(x, model):
    # eq 4 in paper
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    gx_cur = (gx * x).sum(-1)[:, :, None]
    return gx - gx_cur


#%% define EBM classes
# TODO: Try multiple flips per step
# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, latent_shape, n_steps=1, approx=True, temp=2.):
        super().__init__()
        self.dim = np.prod(latent_shape)
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

class EBM(Sampler):
    def __init__(self, H, embedding_weight, buffer=None, mean=None):
        super().__init__(H, embedding_weight)
        self.batch_size = H.batch_size
        self.mcmc_steps = H.mcmc_steps
        self.grad_clip_threshold = H.grad_clip_threshold
        self.all_inds = list(range(H.buffer_size))
        self.buffer = buffer
        self.mean = None if mean is None else nn.Parameter(mean, requires_grad=False)        
        
        self.net = ResNetEBM_cat(H.emb_dim, H.block_str)
        self.gibbs_sampler = DiffSamplerMultiDim(self.latent_shape)

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

    # single training step of the EBM
    # TODO encode input to one hot before processing
    def train_iter(self, x):
        if self.buffer == None:
            raise ValueError('Please set a buffer for the EBM before training')
        stats = {}
        
        buffer_inds = sorted(np.random.choice(self.all_inds, self.batch_size, replace=False))
        x_buffer_ids = self.buffer[buffer_inds]
        x_buffer_ids = x_buffer_ids.cuda()
        x_fake = latent_ids_to_onehot(x_buffer_ids, self.latent_shape, self.codebook_size)
        
        hops = []  # keep track of how much the sampler moves particles around
        for _ in range(self.mcmc_steps):
            # try:
            x_fake_new = self.gibbs_sampler.step(x_fake.detach(), self).detach()
            h = (x_fake_new != x_fake).float().view(x_fake_new.size(0), -1).sum(-1).mean().item()
            hops.append(h)
            x_fake = x_fake_new
            # except ValueError as e:
            #     log(f'Error at step {step}, sampling step {k}: {e}')
            #     log(f'Skipping sampling step and hoping it still works')

        stats['hops'] = (np.mean(hops))
        stats['grad_norm'] = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            self.grad_clip_threshold
        ).item()

        logp_real = self.forward(x).squeeze()
        logp_fake = self.forward(x_fake).squeeze()
        stats['loss'] = logp_fake.mean() - logp_real.mean()

        # update buffer
        self.buffer[buffer_inds] = x_fake.max(2)[1].detach().cpu() 
        
        return stats

    # simply samples from the buffer, AIS sampling during training would be absurd
    def sample(self):
        buffer_inds = sorted(np.random.choice(self.all_inds, self.n_samples, replace=False))
        x_buffer_ids = self.buffer[buffer_inds]
        x_buffer_ids = x_buffer_ids.cuda()
        x_fake = latent_ids_to_onehot(x_buffer_ids, self.latent_shape, self.codebook_size)
        return x_fake

    def class_conditional_train_iter(self, x, y):
        return super().class_conditional_train_iter(x, y)

    def class_conditional_sample(n_samples, y):
        return super().class_conditional_sample(y)