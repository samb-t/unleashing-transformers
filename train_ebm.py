import torch
import visdom
import time
from vqgan import *
from energy import *
from hparams import get_hparams
from utils import *

def main(H):
    data_dim = np.prod(H.latent_shape)
    sampler = DiffSamplerMultiDim(data_dim, 1)

    ae = VQAutoEncoder(
        H.n_channels,
        H.nf,
        H.res_blocks, 
        H.codebook_size, 
        H.emb_dim, 
        H.ch_mult, 
        H.img_size, 
        H.attn_resolutions
    )
    ae = load_model(ae, 'ae', H.ae_load_step, f'vqgan_{H.dataset}_{H.latent_shape[-1]}')

    latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_latents'
    if os.path.exists(latents_filepath):
        latent_ids = torch.load(latents_filepath)
    else:
        full_dataloader = get_data_loader(H.dataset, H.img_size, 5, drop_last=False, shuffle=False)
        ae = ae.cuda() # put ae on GPU for generating
        latent_ids = generate_latent_ids(ae, full_dataloader, H)
        ae = ae.cpu() # put back onto CPU to save memory during EBM training
        save_latents(latent_ids, H.dataset, H.latent_shape[-1])

    latent_loader = torch.utils.data.DataLoader(latent_ids, batch_size=H.ebm_batch_size, shuffle=False)
    latent_iterator = cycle(latent_loader)

    init_dist_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_init_dist'
    if os.path.exists(init_dist_filepath):
        log(f'Loading init distribution from {init_dist_filepath}')
        init_dist = torch.load(init_dist_filepath)
        init_mean = init_dist.mean
    else:  
        # latents # B, H*W, codebook_size
        eps = 1e-3 / H.codebook_size

        batch_sum = torch.zeros(H.latent_shape[1]*H.latent_shape[2], H.codebook_size).cuda()
        log('Generating init distribution:')
        for batch in tqdm(latent_loader):
            batch = batch.cuda()
            latents = latent_ids_to_onehot(batch, H)
            batch_sum += latents.sum(0)

        init_mean = batch_sum / (len(latent_loader) * H.ebm_batch_size)

        init_mean += eps # H*W, codebook_size
        init_mean = init_mean / init_mean.sum(-1)[:, None] # renormalize pdfs after adding eps
        init_dist = MyOneHotCategorical(init_mean.cpu())
        
        torch.save(init_dist, f'latents/{H.dataset}_init_dist')


    # create energy model
    net = ResNetEBM_cat(H.emb_dim, H.block_str)
    energy = EBM(
        net,
        ae.quantize.embedding,
        H.codebook_size,
        H.emb_dim,
        H.latent_shape,
        mean=init_mean,
    ).cuda()
    optim = torch.optim.Adam(energy.parameters(), lr=H.ebm_lr)
    
    # optim = torch.optim.SGD(energy.parameters(), lr=H.ebm_lr, momentum=.9)

    if H.load_step > 0:
        energy = load_model(energy, 'ebm', H.load_step, H.log_dir)
        buffer = load_buffer(H.load_step, H.log_dir)
        
        if H.load_optim:
            optim = load_model(optim, 'ebm_optim', H.load_step, H.log_dir)
            for param_group in optim.param_groups:
                param_group['lr'] = H.ebm_lr
        start_step = H.load_step

    else:
        # HACK
        buffer = []
        for _ in range(int(H.buffer_size / 100)):
            buffer.append(init_dist.sample((100,)).max(2)[1].cpu())
        buffer = torch.cat(buffer, dim=0)

        start_step = 0 

    print('Buffer successfully generated')
    log(f'EBM Parameters: {len(ptv(energy.parameters()))}')

    # check reconstructions are correct
    latent_id_batch = next(latent_iterator)[:64]

    latent_batch = latent_ids_to_onehot(latent_id_batch, H)
    quant = energy.embed(latent_batch.cuda()).cpu()
    with torch.no_grad():
        recons = ae.generator(quant)
    vis.images(recons.clamp(0,1), win='recon_check', opts=dict(title='recon_check'))

    hop_dists = []
    grad_norms = []
    diffs = []
    all_inds = list(range(H.buffer_size))

    # main training loop
    for step in range(start_step, H.train_steps):
        step_time_start = time.time()
        # linearly anneal in learning rate
        if step < H.warmup_iters:
            lr = H.ebm_lr * float(step) / H.warmup_iters
            for param_group in optim.param_groups:
                param_group['lr'] = lr
        
        latent_ids = next(latent_iterator)
        latent_ids = latent_ids.cuda()
        x = latent_ids_to_onehot(latent_ids, H)


        buffer_inds = sorted(np.random.choice(all_inds, H.ebm_batch_size, replace=False))
        x_buffer_ids = buffer[buffer_inds]
        x_buffer = latent_ids_to_onehot(x_buffer_ids, H).cuda()
        x_fake = x_buffer

        hops = []  # keep track of how much the sampler moves particles around
        for k in range(H.sampling_steps):
            try:
                x_fake_new = sampler.step(x_fake.detach(), energy).detach()
                h = (x_fake_new != x_fake).float().view(x_fake_new.size(0), -1).sum(-1).mean().item()
                hops.append(h)
                x_fake = x_fake_new
            except ValueError as e:
                log(f'Error at step {step}, sampling step {k}: {e}')
                log(f'Skipping sampling step and hoping it still works')
                hops.append(0)
        hop_dists.append(np.mean(hops))

        # update buffer
        buffer_ids = x_fake.max(2)[1] 
        buffer[buffer_inds] = buffer_ids.detach().cpu()

        logp_real = energy(x).squeeze()
        logp_fake = energy(x_fake).squeeze()

        obj = logp_real.mean() - logp_fake.mean()

        # L2 regularisation
        loss = -obj + H.l2_coef * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())

        optim.zero_grad()
        loss.backward()

        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(energy.parameters(), H.grad_clip_threshold).item()

        grad_norms.append(grad_norm)
        diffs.append(obj.item())
        step_time_finish = time.time()
        step_time_taken = step_time_finish - step_time_start
        steps = list(range(start_step, step+1))
        vis.line(grad_norms, steps, win='grad_norms', opts=dict(title='Gradient Norms'))
        vis.line(diffs, steps, win='diffs', opts=dict(title='Energy Difference'))
        vis.line(hop_dists, steps, win='hops', opts=dict(title='Hops'))

        if grad_norm > H.grad_clip_threshold:
            log(f'Grad norm breached threshold, skipping step {step}')

        else:     
            optim.step()

            if step % H.steps_per_log == 0:
                log(f"Step: {step}, time: {step_time_taken:.3f}, log p(real)={logp_real.mean():.4f}, log p(fake)={logp_fake.mean():.4f}, diff={obj:.4f}, hops={hop_dists[-1]:.4f}, grad_norm={grad_norm:.4f}")
            if step % H.steps_per_display_samples == 0:
                q = energy.embed(x_fake).cpu()
                with torch.no_grad():
                    samples = ae.generator(q)
                vis.images(samples[:64].clamp(0,1), win='samples', opts=dict(title='samples'))
                
                if step % H.steps_per_save_samples == 0:
                    save_images(samples[:64], vis, 'samples', step, H.log_dir)
            if step % H.steps_per_ebm_checkpoint == 0 and step > 0 and not (H.load_step == step):
                save_model(energy, 'ebm', step, H.log_dir)
                save_model(optim, 'ebm_optim', step, H.log_dir)
                save_buffer(buffer, step, H.log_dir)

if __name__ == '__main__':
    H = get_hparams()
    vis = setup_visdom(H)
    if H.ae_load_step:
        config_log(H.log_dir)
        start_training_log(H.get_ebm_param_dict())
        main(H)
    else:
        print(f'Please select an autoencoder to load using the --ae_load_step flag')