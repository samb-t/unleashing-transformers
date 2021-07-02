import torch
import time
import math
from unet import SegmentationUnet
from energy import generate_latent_ids, latent_ids_to_onehot
from vqgan import *
from multinomial_diffusion import *
from hparams_bert import get_hparams
from utils import *


def loglik_bpd(model, x):
    """Compute the log-likelihood in bits per dim."""
    return - model.log_prob(x).sum() / (math.log(2) * x.shape.numel())


def main(H):

    ae = VQAutoEncoder(
        H.n_channels,
        H.nf,
        H.res_blocks, 
        H.codebook_size, 
        H.emb_dim, 
        H.ch_mult, 
        H.img_size, 
        H.attn_resolutions
    ).cuda()
    ae = load_model(ae, 'ae', H.ae_load_step, f'vqgan_{H.dataset}_{H.latent_shape[-1]}')

    latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_latents'
    if os.path.exists(latents_filepath):
        latent_ids = torch.load(latents_filepath)
    else:
        full_dataloader = get_data_loader(H.dataset, H.img_size, H.vqgan_bs, drop_last=False, shuffle=False)
        ae = ae.cuda() # put ae on GPU for generating
        latent_ids = generate_latent_ids(ae, full_dataloader, H)
        # ae = ae.cpu() # put back onto CPU to save memory during EBM training
        save_latents(latent_ids, H.dataset, H.latent_shape[-1])

    print(torch.unique(latent_ids).shape)

    latent_loader = torch.utils.data.DataLoader(latent_ids, batch_size=H.md_batch_size, shuffle=False)
    latent_iterator = cycle(latent_loader)

    unet = SegmentationUnet(
        num_classes=H.codebook_size,
        dim=32,
        num_steps=1000,
        dim_mults=(1,2,4,8),
        dropout=0.
    )

    # create multinomial diffusion model
    diffusion = MultinomialDiffusion(
        H.codebook_size,
        tuple(H.latent_shape),
        unet,
        ae.quantize.embedding,
        H.latent_shape,
        H.emb_dim,
        timesteps=1000,
        loss_type='vb_stochastic'
    ).cuda()
    optim = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    
    # optim = torch.optim.SGD(energy.parameters(), lr=H.ebm_lr, momentum=.9)

    start_step = 0
    if H.load_step > 0:
        diffusion = load_model(diffusion, 'diffusion', H.load_step, H.log_dir)
        
        if H.load_optim:
            optim = load_model(optim, 'ebm_optim', H.load_step, H.log_dir)
            for param_group in optim.param_groups:
                param_group['lr'] = H.ebm_lr
        start_step = H.load_step

    log(f'Diffusion Parameters: {len(ptv(diffusion.parameters()))}')

    # check reconstructions are correct
    latent_id_batch = next(latent_iterator)[:64]

    latent_batch = latent_ids_to_onehot(latent_id_batch, H)
    quant = diffusion.embed(latent_batch.cuda())
    with torch.no_grad():
        recons = ae.generator(quant)
    vis.images(recons.clamp(0,1), win='recon_check', opts=dict(title='recon_check'))

    losses = np.array([])
    loss_means = np.array([])
    # main training loop
    for step in range(start_step, H.train_steps):
        step_time_start = time.time()
        
        x = next(latent_iterator) # latent ids - bs x 1 x latent_width x latent_width
        x = x.reshape(-1, H.latent_shape[-2], H.latent_shape[-1]).cuda()

        loss = loglik_bpd(diffusion, x)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        losses = np.append(losses, loss.item())

        step_time_finish = time.time()
        step_time_taken = step_time_finish - step_time_start

        if step % H.steps_per_log == 0:
            log(f"Step: {step}, time: {step_time_taken:.3f}, log lkhd={loss.item():.4f}")
            loss_means = np.append(loss_means, np.mean(losses))
            losses = np.array([])
            vis.line(loss_means, [0] + list(range(start_step, step, H.steps_per_log)), win='loss', opts=dict(title='NLL'))
        if step % H.steps_per_display_samples == 0 and step > 0:
            q = diffusion.sample_chain(H.md_batch_size)[0]
            q = q.reshape(H.md_batch_size, 1, -1)
            q = latent_ids_to_onehot(q, H)
            q = diffusion.embed(q.cuda())
            with torch.no_grad():
                samples = ae.generator(q)
            vis.images(samples[:64].clamp(0,1), win='samples', opts=dict(title='samples'))
            
            if step % H.steps_per_save_samples == 0:
                save_images(samples[:64], 'samples', step, H.log_dir)
        if step % H.steps_per_md_checkpoint == 0 and step > 0 and not (H.load_step == step):
            save_model(diffusion, 'md', step, H.log_dir)
            save_model(optim, 'md_optim', step, H.log_dir)

if __name__ == '__main__':
    H = get_hparams()
    vis = setup_visdom(H)
    if H.ae_load_step:
        config_log(H.log_dir)
        start_training_log(H.get_bert_param_dict())
        main(H)
    else:
        print(f'Please select an autoencoder to load using the --ae_load_step flag')