import os
import torch
from tqdm import tqdm
from .log_utils import save_latents, log

@torch.no_grad()
def get_samples(H, generator, sampler):            
    #TODO need to write sample function for EBM (give option of AIS?)
    sampler.eval()
    latents = sampler.sample() 
    latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size)
    if H.deepspeed:
        if H.deepspeed:
            latents_one_hot = latents_one_hot.half()
        latents_one_hot = latents_one_hot.cuda()
    q = sampler.embed(latents_one_hot)
    images = generator(q.float()) # move to cpu if not keeping generator on GPU
    sampler.train()

    return images


def unpack_sampler_stats(stats):
    return (
        stats['losses'],
        stats['mean_losses'],
        stats['steps_per_log']
    )


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
def get_init_mean(H, latent_loader):
    init_mean_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_init_mean'
    if os.path.exists(init_mean_filepath):
        log(f'Loading init distribution from {init_mean_filepath}')
        init_mean = torch.load(init_mean_filepath)
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
        
        torch.save(init_mean, f'latents/{H.dataset}_{H.latent_shape[-1]}_init_mean')

    return init_mean


@torch.no_grad()
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
    save_latents(latent_ids_out, H.dataset, H.latent_shape[-1])


@torch.no_grad()
def get_latent_loader(H, latents_filepath):
    latent_ids = torch.load(latents_filepath)
    latent_loader = torch.utils.data.DataLoader(latent_ids, batch_size=H.batch_size, shuffle=True, pin_memory=True)    
    return latent_loader


def retrieve_autoencoder_components_state_dicts(H, components_list, remove_component_from_key=False):
    state_dict = {}

    # default to loading ema models first
    ae_load_path = f'logs/{H.ae_load_dir}/saved_models/vqgan_ema_{H.ae_load_step}.th'        
    if not os.path.exists(ae_load_path):
        ae_load_path = f'logs/{H.ae_load_dir}/saved_models/vqgan_{H.ae_load_step}.th'        
    
    full_vqgan_state_dict = torch.load(ae_load_path, map_location='cpu')

    for key in full_vqgan_state_dict:
        for component in components_list:
            if component in key:
                new_key = key[3:] # remove 'ae.'
                if remove_component_from_key:
                    new_key = new_key[len(component)+1:] # e.g. remove 'quantize.'

                state_dict[new_key] = full_vqgan_state_dict[key]            

    return state_dict