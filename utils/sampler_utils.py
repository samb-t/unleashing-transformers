import os
import torch
from tqdm import tqdm
from .log_utils import save_latents, display_images, log
from .data_utils import get_data_loader

@torch.no_grad()
def display_samples(H, vis, generator, sampler):            

    #TODO need to write sample function for EBMS (give option of AIS?)
    latents = sampler.sample() 
    latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size)
    if H.deepspeed:
        latents = latents.half()
    q = sampler.embed(latents)
    images = generator(q.cpu().float())
    display_images(vis, images, H, win_name=f'{H.model}_samples')

    return None


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


# TODO: review this code
def get_init_mean(H, latent_loader, cuda=False):
    init_dist_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_init_mean'
    if os.path.exists(init_dist_filepath):
        log(f'Loading init distribution from {init_dist_filepath}')
        init_dist = torch.load(init_dist_filepath)
    else:  
        # latents # B, H*W, codebook_size
        eps = 1e-3 / H.codebook_size

        batch_sum = torch.zeros(H.latent_shape[1]*H.latent_shape[2], H.codebook_size).cuda()
        log('Generating init distribution:')
        for batch in tqdm(latent_loader):
            batch = batch.cuda()
            latents = latent_ids_to_onehot(batch, H.latent_shape, H.codebook_size)
            batch_sum += latents.sum(0)

        init_mean = batch_sum / (len(latent_loader) * H.batch_size)

        init_mean += eps # H*W, codebook_size
        init_mean = init_mean / init_mean.sum(-1)[:, None] # renormalize pdfs after adding eps
        if cuda:
            init_dist = MyOneHotCategorical(init_mean.cuda())
        else:
            init_dist = MyOneHotCategorical(init_mean.cpu())
        
        torch.save(init_dist, f'latents/{H.dataset}_{H.latent_shape[-1]}_init_dist')

    return init_dist


def generate_latent_ids(H, ae, dataloader):
    latent_ids = []
    for x, _ in tqdm(dataloader):
        x = x.cuda()
        z = ae.encoder(x) # B, emb_dim, H, W

        z = z.permute(0, 2, 3, 1).contiguous() # B, H, W, emb_dim
        z_flattened = z.view(-1, H.emb_dim) # B*H*W, emb_dim

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + \
            (ae.quantize.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, ae.quantize.embedding.weight.t())
        
        min_encoding_indices = torch.argmin(d, dim=1)

        latent_ids.append(min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous())

    latent_ids_out = torch.cat(latent_ids, dim=0)

    return latent_ids_out


def get_latent_loader(H, ae):
    latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_latents'
    if os.path.exists(latents_filepath):
        latent_ids = torch.load(latents_filepath)
    else:
        full_dataloader = get_data_loader(H.dataset, H.img_size, H.vqgan_batch_size, drop_last=False, shuffle=False)
        ae = ae.cuda() # put ae on GPU for generating
        latent_ids = generate_latent_ids(H, ae, full_dataloader)
        ae = ae.cpu() # put back onto CPU to save memory during EBM training
        save_latents(latent_ids, H.dataset, H.latent_shape[-1])

    latent_loader = torch.utils.data.DataLoader(latent_ids, batch_size=H.batch_size, shuffle=True)
    
    return latent_loader