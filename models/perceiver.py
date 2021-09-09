import math
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import wraps

# perceiver but the last layer (or n before layer layer) uses a learned query 
# the size of the original input where K and V are obtained from the normal
# perceiver output.
#
# sort of like an hourglass shape.
#
# Unlike linformer never compresses the input so we always have access to the full
# original input.
#
#
# Ideas:
# - could in theory use different queries for different time steps
#   (or some set of queries e.g. first queries are used for the first 50 steps, etc.)
#
# Justification: when the number of masked elements is reasonably small, its not like we
# actually need output the same size output as input. So a smaller number of queries could
# be much more economical.

# This is just PerceiverIO :')

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class PerceiverIO(nn.Module):
    def __init__(self, H, cross_heads=1, weight_tie_layers=False, decoder_ff=True):
        super().__init__()
        latent_dim, latent_heads = H.bert_n_emb, H.bert_n_head
        latent_dim_head = H.perceiver_dim_head
        dim, cross_dim_head = H.bert_n_emb, H.perceiver_dim_head
        queries_dim, logits_dim = H.bert_n_emb, H.codebook_size
        depth, self_per_cross_attn = H.perceiver_layers, H.layers_per_cross_attn
        # Perceiever IO uses H.perceiver_layers=1, H.self_per_cross_attn=n

        self.time_chunks = H.diffusion_steps / H.perceiver_latent_chunks
        self.perceiver_latent_chunks = H.perceiver_latent_chunks
        
        self.latents = nn.Parameter(torch.randn(H.perceiver_latent_chunks, H.perceiver_latents, H.bert_n_emb))
        self.output_queries = nn.Parameter(torch.randn(H.perceiver_latent_chunks, np.prod(H.latent_shape), H.bert_n_emb))
        self.tok_emb = nn.Embedding(H.codebook_size+1, dim)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, H.block_size, dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))
        
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    # when mask is passed in it can't learn at all. Check mask isn't inverted. If fine then it must need
    # to know where the masked tokens are, which makes sense, allows it to work out where it needs 
    # to find out about
    def forward(self, idx, t=None, mask=None):
        time = t
        # TODO: Try time variable latents
        token_embeddings = self.tok_emb(idx)

        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]

        data = token_embeddings + position_embeddings

        time = (time / self.time_chunks).long().clamp(0,self.perceiver_latent_chunks-1)

        x = self.latents[time]
        # x = repeat(self.latents, 'n d -> b n d', b=data.size(0))

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x
        
        # output_queries = repeat(self.output_queries, 'n d -> b n d', b=data.size(0))
        output_queries = self.output_queries[time]
        x = self.decoder_cross_attn(output_queries, context = x)
        x = self.decoder_ff(x) + x

        return self.to_logits(x)


# 24 layer 512D 16 heads each 64D is a 167M param model and with a batch size of 48 
# uses around 10gb of memory, taking around 0.22s per step.

# What we should try is a standard transformer, 512D, 16 heads each 64D too.