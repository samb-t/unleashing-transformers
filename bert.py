import math
import random
import time
import numpy as np
import torch
from torch import sparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, latent_size, embedding, latent_emb_dim,
                 n_layer=12, n_head=8, n_embd=256, embd_pdrop=0., 
                 resid_pdrop=0., attn_pdrop=0.):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size-1, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.latent_size = latent_size
        self.embedding = embedding
        self.embedding.requires_grad = False
        self.latent_emb_dim = latent_emb_dim
    
    def embed(self, z):
        z_flattened = z.view(-1, self.vocab_size-1) # B*H*W, codebook_size
        return torch.matmul(z_flattened, self.embedding.weight).view(
            z.size(0), 
            self.latent_size[1],
            self.latent_size[2],
            self.latent_emb_dim
        ).permute(0, 3, 1, 2).contiguous()

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
        
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # reduce so that each batch item is weighted equally (some will have more -1 items than others)
            loss = F.cross_entropy(logits.permute(0,2,1), targets, ignore_index=-1, reduction='none')
            num_trainable = (targets != -1).float().sum(1) # number of elements used to train
            loss = torch.mean(loss.sum(1) / num_trainable) # mean down batch

        return logits, loss

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, latent_ids, mask_id, num_ids):
        self.latent_ids = latent_ids
        self.mask_id = mask_id
        self.num_ids = num_ids
    
    def __len__(self):
        return self.latent_ids.size(0)
    
    def __getitem__(self, item):
        latent = self.latent_ids[item].clone()
        target = []

        num_to_mask = random.randint(1, latent.size(0))
        mask = np.random.choice(np.arange(latent.size(0)), size=num_to_mask, replace=False)

        # NOTE: When calculating loss make sure to mean over dim 1 then over dim 0.

        for idx, l in enumerate(latent):
            if idx in mask: # mask token
                target.append(l.clone())
                latent[idx] = self.mask_id 
            elif random.random() < 0.02: # randomly change to random token
                target.append(l.clone())
                latent[idx] = random.randrange(self.num_ids)
            else:
                target.append(-1)

            # prob = random.random()
            # if prob < 0.15:
            #     prob /= 0.15
            #     target.append(l.clone())

            #     if prob < 0.8: # 80% randomly change token to mask token
            #         latent[idx] = self.mask_id
            #     elif prob < 0.9: # 10% randomly change to random token
            #         latent[idx] = random.randrange(self.num_ids)
            #     # 10% randomly don't change but use to train network
                
            # else:
            #     # mark to not train with
            #     target.append(-1)
        
        target = torch.tensor(target).reshape(latent.shape).long()
        
        return latent, target

@torch.no_grad()
def warm_start_from_real(model, mask_id, data_dim, batch_size=32, latents=None):

    if latents == None:
        latents = torch.ones(batch_size, data_dim).long().cuda() * mask_id

    model.eval()
    for k in range(data_dim): # greedy decode
        latents[:,k] = mask_id
        logits, _ = model(latents)

        probs = F.softmax(logits[:,k,:], dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        # ix = logits[:,k,:].max(1)[1].unsqueeze(-1)
        latents[:,k] = ix[:,0]

    model.train()

    return latents


@torch.no_grad()
def MH_sampling(model, mask_id, data_dim, batch_size=32, mcmc_steps=50, energy_type='norm', eps=1e-8):
    latents = torch.ones(batch_size, data_dim).long().cuda() * mask_id

    energy_prev = torch.zeros(latents.size(0), 1)
    # energy_prev = implicit_energy_fn(model, latents, mask_id, energy_type=energy_type) # bs x 1

    # TODO: force changes to occur - don't allow reproducing same token 
    acceptance_rate = 0
    all_acceptance_rates = []
    for i in tqdm(range(mcmc_steps)):
        block_size = data_dim - int((data_dim / mcmc_steps) * i)

        block_start_index = random.randint(0, latents.size(1)-block_size) 
        current_x_i = latents[:, block_start_index : block_start_index + block_size].clone() # bs x block_size
        proposal_latents = latents.clone()
        proposal_latents[:, block_start_index : block_start_index + block_size] = mask_id 
        logits, _ = model(proposal_latents) # bs x latent_len x codebook_size
        
        q_prop_x = 0
        q_x_prop = 0

        # log softmax?
        block_probs = logits[:, block_start_index:block_start_index+block_size,:] # bs x block_size x codebook_size
        for idx, probs_index in enumerate(range(block_start_index, block_start_index+block_size)):
            probs = F.softmax(block_probs[:, idx], dim=1) # bs x codebook_size
            proposal_tokens = torch.multinomial(probs, num_samples=1) # bs x 1
            proposal_latents[:, probs_index] = proposal_tokens.squeeze(1)

            if i > 0: # tries to gather probabilities for masked inputs on the first step
                log_probs = F.log_softmax(block_probs[:,idx], dim=1)
                q_prop_x += torch.gather(log_probs, 1, proposal_tokens) # bs x 1
                q_x_prop += torch.gather(log_probs, 1, current_x_i[:, idx].unsqueeze(1)) # bs x 1

        energy_proposal = implicit_energy_fn(model, proposal_latents, mask_id, energy_type=energy_type) # bs x 1

        if i == 0: # always accept the first one
            energy_prev = energy_proposal
            latents = proposal_latents
            continue

        # check energy should definitely be log-ed. I have a feeling it shouldn't be but really not sure
        acceptance_prob = torch.exp(energy_proposal + q_x_prop - energy_prev - q_prop_x)

        # print(energy_proposal, q_x_prop, energy_prev, q_prop_x, acceptance_prob)

        # + eps for stability. If exp(q_x_prop) is 0 then this division == 0 rather than 1
        # acceptance_prob = (energy_proposal * (torch.exp(q_x_prop) + eps)) / (energy_prev * (torch.exp(q_prop_x) + eps)) 
        acceptance_prob = acceptance_prob.clamp(max=1)
        acceptance_mask = torch.rand_like(acceptance_prob) <= acceptance_prob # bs x 1

        acceptance_rate += acceptance_mask.float().mean()

        all_acceptance_rates.append(acceptance_mask.float().mean().item())

        energy_prev[acceptance_mask] = energy_proposal[acceptance_mask]
        latents[acceptance_mask.squeeze(1)] = proposal_latents[acceptance_mask.squeeze(1)]
        
    acceptance_rate /= mcmc_steps

    return latents, acceptance_rate, all_acceptance_rates

@torch.no_grad()
def annealed_block_sampling(model, mask_id, data_dim, batch_size=32, mcmc_steps=50, energy_type='norm'):
    latents = torch.ones(batch_size, data_dim).long().cuda() * mask_id

    for i in range(mcmc_steps):
        num_to_mask = data_dim - int((data_dim / mcmc_steps) * i)
        sparse_mask = torch.stack([torch.tensor(np.random.choice(np.arange(data_dim), size=num_to_mask, replace=False)) for _ in range(latents.size(0))], dim=0)

        mask = torch.zeros_like(latents) != 0
        mask.scatter_(1, sparse_mask, torch.zeros_like(latents) == 0)

        proposed_latents = latents.clone()
        proposed_latents[mask] - mask_id

@torch.no_grad()
def implicit_energy_fn(model, latents, mask_id, energy_type='norm'):
    energy_total = 0
    for i in range(latents.size(1)):
        current_x_i = latents[:, i].clone()
        latents[:, i] = mask_id
        logits, _ = model(latents)
        if energy_type == 'norm':
            probs = F.log_softmax(logits[:,i,:], dim=-1) # bs x codebook_size

        energy_total += torch.gather(probs, 1, current_x_i.unsqueeze(1))
        latents[:, i] = current_x_i
        
    return energy_total # bs x 1
