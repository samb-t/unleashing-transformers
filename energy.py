#%% imports
import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
from vqgan import VQAutoEncoder, ResBlock, VectorQuantizer, Encoder, Generator
from tqdm import tqdm
import os
import visdom
from utils import *
from torch.nn.utils import parameters_to_vector as ptv

#%% hparams
dataset = 'flowers'
if dataset == 'mnist':
    batch_size = 128
    img_size = 32
    n_channels = 1
    codebook_size = 10
    emb_dim = 64
    buffer_size = 10000
    sampling_steps = 50
    warmup_iters = 2000
    main_lr = 1e-4
    latent_shape = [1, 8, 8]
elif dataset == 'cifar10':
    batch_size = 128
    img_size = 32
    n_channels = 3
    codebook_size = 128
    emb_dim = 256
    buffer_size = 10000
    sampling_steps = 50
    warmup_iters = 2000
    main_lr = 1e-4
    latent_shape = [1, 8, 8]
elif dataset == 'flowers':
    batch_size = 128
    img_size = 32
    n_channels = 3
    codebook_size = 128
    emb_dim = 128
    buffer_size = 1000
    sampling_steps = 50
    warmup_iters = 2000
    main_lr = 1e-4
    latent_shape = [1, 8, 8]

training_steps = 100001
steps_per_log = 10
steps_per_eval = 100
steps_per_checkpoint = 500
grad_clip_threshold = 1000

LOAD_MODEL = False
LOAD_MODEL_STEP = 0

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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*[BasicBlock(emb_dim, emb_dim, 1) for _ in range(6)])
        self.energy_linear = nn.Linear(emb_dim, 1)

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
        z_flattened = z.view(-1, codebook_size) # B*H*W, codebook_size
        return torch.matmul(z_flattened, self.embedding.weight).view(z.size(0), latent_shape[1], latent_shape[2], emb_dim).permute(0, 3, 1, 2).contiguous()

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


def get_latents(ae, dataloader):
    latents = []
    for x, _ in tqdm(dataloader):
        x = x.cuda()
        z = ae.encoder(x) # B, 64, H, W

        z = z.permute(0, 2, 3, 1).contiguous() # B, H, W, 64
        z_flattened = z.view(-1, emb_dim) # B*H*W, 64

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (ae.quantize.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, ae.quantize.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], ae.quantize.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        one_hot = min_encodings.view(z.size(0), latent_shape[1], latent_shape[2], codebook_size)

        latents.append(one_hot.reshape(one_hot.size(0), -1, codebook_size).cpu().contiguous())
    
    return torch.cat(latents, dim=0)


def main():
    data_dim = np.prod(latent_shape)
    sampler = DiffSamplerMultiDim(data_dim, 1)

    ae = VQAutoEncoder(n_channels, emb_dim, codebook_size).cuda()
    ae = load_model(ae, 'ae', 300, f'vq_gan_test_{dataset}')

    if os.path.exists(f'latents/{dataset}_latents.pkl'):
        latents = torch.load(f'latents/{dataset}_latents.pkl')
    else:
        full_dataloader = get_data_loader(dataset, img_size, batch_size, drop_last=False)
        latents = get_latents(ae, full_dataloader)
        save_latents(latents, dataset)

    data_iterator = cycle(torch.utils.data.DataLoader(latents, batch_size=batch_size, shuffle=True))

    # latents # B, H*W, 10
    eps = 1e-3 / codebook_size
    init_mean = latents.mean(0) + eps # H*W, 10
    init_mean = init_mean / init_mean.sum(-1)[:, None] # renormalize pdfs after adding eps

    init_dist = MyOneHotCategorical(init_mean)
    buffer = init_dist.sample((buffer_size,)) # 1000, H*W, 10
    all_inds = list(range(buffer_size))

    # create energy model
    net = ResNetEBM_cat()
    energy = EBM(net, ae.quantize.embedding, mean=init_mean).cuda() # 10x64
    optim = torch.optim.Adam(energy.parameters())

    start_step = 0 
    if LOAD_MODEL:
        energy = load_model(energy, 'ebm', LOAD_MODEL_STEP, log_dir)
        optim = load_model(optim, 'ebm_optim', LOAD_MODEL_STEP, log_dir)
        start_step = LOAD_MODEL_STEP

    log(f'EBM Parameters: {len(ptv(energy.parameters()))}')

    # check reocnstructions are correct
    latent_batch = next(data_iterator)[:64].cuda()
    quant = energy.embed(latent_batch)
    recons = ae.generator(quant)
    vis.images(recons.clamp(0,1), win='recon_check', opts=dict(title='recon_check'))

    hop_dists = []

    grad_norms = []

    #%% main training loop
    for step in range(start_step, training_steps):
        # linearly anneal in learning rate
        if step < warmup_iters:
            lr = main_lr * float(step) / warmup_iters
            for param_group in optim.param_groups:
                param_group['lr'] = lr
        
        x = next(data_iterator)
        x = x.cuda()

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

        # gradient clipping / tracking
        grad_norm = torch.nn.utils.clip_grad_norm_(energy.parameters(), grad_clip_threshold).item()
        grad_norms.append(grad_norm)

        optim.step()

        if step % steps_per_log == 0:
            log(f"Step: {step}, log p(real)={logp_real.mean():.4f}, log p(fake)={logp_fake.mean():.4f}, diff={obj:.4f}, hops={hop_dists[-1]:.4f}")
            q = energy.embed(x_fake)
            samples = ae.generator(q)
            vis.images(samples[:64].clamp(0,1), win='samples', opts=dict(title='samples'))
            vis.line(grad_norms, win='grad_norms', opts=dict(title='Gradient Norms'))

        if step % steps_per_eval == 0:
            q = energy.embed(x_fake)
            samples = ae.generator(q)
            save_images(samples[:64], vis, 'samples', step, log_dir)

        if step % steps_per_checkpoint == 0 and step > 0:
            save_model(energy, 'ebm', step, log_dir)
            save_model(optim, 'ebm_optim', step, log_dir)

if __name__ == '__main__':
    vis = visdom.Visdom()
    log_dir = f'ebm_{dataset}'
    config_log(log_dir)
    start_training_log(dict(
        dataset = dataset,
        batch_size = batch_size,
        img_size = img_size,
        n_channels = n_channels,
        codebook_size = codebook_size,
        emb_dim = emb_dim,
        buffer_size = buffer_size,
        sampling_steps = sampling_steps,
        warmup_iters = warmup_iters,
        main_lr = main_lr,
        latent_shape = latent_shape
    ))
    main()