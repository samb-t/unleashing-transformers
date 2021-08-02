import torch
import torch.nn.functional as F
import torch.distributions as dists
import numpy as np
import math
from .sampler import Sampler


class AbsorbingDiffusion(Sampler):
    def __init__(self, H, denoise_fn, mask_id, embedding_weight, aux_weight=0.01):
        super().__init__(H, embedding_weight=embedding_weight)

        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = H.diffusion_steps
        self.mask_id = mask_id
        self._denoise_fn = denoise_fn
        self.n_samples = H.batch_size

        self.aux_weight = aux_weight
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
    
    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError
        
    def q_sample(self, x_0, t):
        # randomly set token to mask with probability t/T
        x_t, x_0_ignore = x_0.clone(), x_0.clone()
        mask = torch.rand_like(x_t.float()) < (t.float().unsqueeze(-1) / self.num_timesteps)
        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore
    
    def _train_loss(self, x_0):
        b, device = x_0.size(0), x_0.device

        t, pt = self.sample_time(b, device, 'importance')

        x_t, x_0_ignore = self.q_sample(x_0=x_0, t=t)
        x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0,2,1)

        # ELBO: weighted by 1/t as on average there should be t masked tokens
        vb_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1) / t
        # basically just a reweighted variational bound that also takes into account non-masked points
        aux_loss = F.cross_entropy(x_0_hat_logits, x_0, reduction='none').sum(1)

        Lt2 = vb_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # not sure about aux_weighting, best method probably to compare output magnitudes and also just play around
        loss = vb_loss #+ self.aux_weight * aux_loss
        loss = loss / pt
        loss = loss / (math.log(2) * x_0.shape[1:].numel())

        return loss.mean(), vb_loss.mean(), self.aux_weight * aux_loss.mean()
    
    def sample(self):
        b, device = self.n_samples, 'cuda'
        x_0 = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
        for t in reversed(range(1, self.num_timesteps+1)):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, _ = self.q_sample(x_0, t)
            x_0_logits = self._denoise_fn(x_t, t=t)
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0[x_t == self.mask_id] = x_0_hat[x_t == self.mask_id]
            # print("x0 at", t, x_0, x_0.shape)

        return x_0

    def train_iter(self, x):
        loss, vb_loss, aux_loss = self._train_loss(x)
        norm = 1
        stats = {'loss': loss / norm, 'vb_loss': vb_loss, 'aux_loss': aux_loss}
        return stats