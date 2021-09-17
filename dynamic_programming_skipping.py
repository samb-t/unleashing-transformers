from utils.data_utils import get_data_loader
import lpips
import torch
from models import VQGAN
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts,\
                                get_samples, get_latent_loaders 
from utils.log_utils import log, setup_visdom, config_log, start_training_log, \
                             load_model, save_images, display_images
from utils.vqgan_utils import unpack_vqgan_stats, load_vqgan_from_checkpoint, calc_FID
from train_sampler import get_sampler
from tqdm import tqdm
import torchvision
import copy
import math

def main(H, vis):
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight')
    embedding_weight = embedding_weight.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()
    sampler = load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir).cuda()

    latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_train_latents'
    train_latent_loader, val_latent_loader = get_latent_loaders(H, latents_filepath)
    data_loader, _ = get_data_loader(H.dataset, H.img_size, H.batch_size, shuffle=False)
    train_latent_loader = iter(train_latent_loader)

    T = H.diffusion_steps
    L = torch.full((T+1, T+1), float('inf'))
    num_batches = 1

    # Compute KL cost table L
    # for batch_idx in range(num_batches):
    #     z = next(train_latent_loader).cuda()
    #     for t in tqdm(range(1, T+1)):
    #         s = torch.arange(0, t)
    #         l = sampler.elbo_step_unweighted(z, t)
    #         kld = (1 - (s/t)) * l.cpu()
    #         # print(t, kld)
    #         if batch_idx == 0:
    #             L[t,:t] = kld
    #         else:
    #             L[t,:t] += kld
    
    # L = L / num_batches

    # torch.save(L, "loss_table_backup.pkl")
    L = torch.load("loss_table_backup.pkl")
    L[torch.eye(L.size(0)).bool()] = 0.0

    # Algorithm 1
    D = torch.full((T+1, T+1), -1)
    C = torch.full((T+1, T+1), float('inf'))
    C[0,0] = 0

    for k in range(1, T+1):
        bpds = C[k-1,None] + L[:,:] # k or k-1?
        min_values, min_indices = torch.min(bpds, dim=-1)
        # print(min_values)
        C[k] = min_values
        D[k] = min_indices

    # for k in range(1, T+1):
    #     for t in range(T+1):
    #         bpds = C[k-1] + L[t]
    #         min_values, min_indices = torch.min(bpds, dim=0)
    #         C[k,t] = min_values.item()
    #         D[k,t] = min_indices.item()

    print(C)
    print(D)

    print("--------------------------")

    # TODO: Try K=500 or so with infs to force exactly 500 steps.
    # If better than 256 then we know that infs should be used.

    # fetch shortest path of K steps
    K = 50
    optpath = []
    t = torch.tensor(T)
    # cost = 0
    for k in reversed(range(1,K)):
        optpath.append(t.item())
        t = D[k,t]
        print(C[k,t])

    cost = C[K,T]

    print(optpath)
    print(cost)
    print(cost / (256 * math.log(2)))

    print([L[optpath[i],optpath[i+1]].item() for i in range(len(optpath)-1)])
    print(sum([L[optpath[i],optpath[i+1]].item() for i in range(len(optpath)-1)]))
    print(sum([L[optpath[i],optpath[i+1]].item() for i in range(len(optpath)-1)]) / (256 * math.log(2)))


####################################
# TODO: use much bigger batch size #
####################################

if __name__=='__main__':
    H = get_sampler_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')   
    start_training_log(H)
    main(H, vis)