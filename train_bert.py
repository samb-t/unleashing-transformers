import torch
from torchvision import transforms
import visdom
import time
from vqgan import *
from bert import *
from energy import *
from hparams_bert import get_hparams
from utils import *
def main(H):
    data_dim = np.prod(H.latent_shape)


    # TODO: set up to generate latents on GPU
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

    if os.path.exists(f'latents/{H.dataset}_{H.latent_shape[-1]}_latents'):
        latent_ids = torch.load(f'latents/{H.dataset}_{H.latent_shape[-1]}_latents')
    else:
        full_dataloader = get_data_loader(H.dataset, H.img_size, 4, drop_last=False, shuffle=False)

        # latents = get_latents(ae, full_dataloader)
        # save_latents(latents, dataset)
        latent_ids = generate_latent_ids(ae, full_dataloader, H)
        save_latents(latent_ids, H.dataset, H.ae_load_step)
    
    print("latent min", latent_ids.min(), "latent max", latent_ids.max())
    print(torch.unique(latent_ids).shape)

    unmasked_latents = latent_ids[:32].clone()
    
    latent_dataset = BERTDataset(latent_ids.reshape(latent_ids.size(0), -1), H.codebook_size, H.codebook_size)
    latent_loader = torch.utils.data.DataLoader(latent_dataset, batch_size=H.bert_batch_size, shuffle=False)
    latent_iterator = cycle(latent_loader)

    # Simple Transformer
    transformer = GPT(
        H.codebook_size+1, 
        H.block_size, 
        H.latent_shape, 
        ae.quantize.embedding,
        H.emb_dim, 
        n_layer=H.bert_n_layers, 
        n_head=H.bert_n_head, 
        n_embd=H.bert_n_emb
    ).cuda()
    optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    start_step = 0
    if H.load_step > 0:
        transformer = load_model(transformer, 'transformer', H.load_step, H.log_dir)
        if H.load_optim:
            optim = load_model(optim, 'transformer_optim', H.load_step, H.log_dir)
        start_step = H.load_step

    log(f'Transformer Parameters: {len(ptv(transformer.parameters()))}')
    
    nlls = np.array([])
    nll_means = np.array([])
    for step in range(start_step, H.train_steps):
        step_time_start = time.time()
        latent_ids, target = next(latent_iterator)
        latent_ids = latent_ids.cuda()
        target = target.cuda()

        logits, nll = transformer(latent_ids, targets=target)

        optim.zero_grad()
        nll.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1000)

        if torch.isnan(grad_norm):
            log(f"Skipping step {step} with nan gradient")
            continue

        optim.step()
        
        nlls = np.append(nlls, nll.item())
        # TODO: Print accuracy
        accuracy = (logits.max(-1)[1][target > 0] == target[target > 0]).float().mean()

        step_time_finish = time.time()
        step_time_taken = step_time_finish - step_time_start

        # display + save less frequently than ebm
        if step % H.steps_per_log == 0 and step > 0:
            log(f"Step: {step}, time: {step_time_taken:.3f}, nll={nll:.4f}, grad_norm={grad_norm:.4f}, accuracy={accuracy:.4f}")
            nll_means = np.append(nll_means, nlls.mean())
            nlls = np.array([])
            vis.line(nll_means, list(range(start_step, step, H.steps_per_log)), win='nll', opts=dict(title='NLL'))

        if step % H.steps_per_display_samples == 0 and step > 0:

            log('Sampling latents...')
            samples, acceptance_rate, all_acceptance_rates = MH_sampling(transformer, H.codebook_size, data_dim, mcmc_steps=100)
            log(f'Samples generated, acceptance rate: {acceptance_rate*100}%')

            # vis.line(all_acceptance_rates, win='acceptance_rates', opts=dict(title='Acceptance Rates'))
            print(all_acceptance_rates)

            log('Generating images from samples latents...')
            with torch.no_grad():
                q = transformer.embed(latent_ids_to_onehot(samples.reshape(-1, H.latent_shape[-1], H.latent_shape[-1]).contiguous(), H))
                samples = ae.generator(q.cpu())

            vis.images(samples[:64].clamp(0,1), win='samples', opts=dict(title='Samples'))
             
            if step % H.steps_per_save_samples == 0:
                save_images(samples, 'samples', step, H.log_dir)

        if step % H.steps_per_bert_checkpoint == 0 and step > H.load_step:
            save_model(transformer, 'transformer', step, H.log_dir)
            save_model(optim, 'transformer_optim', step, H.log_dir)


if __name__ == '__main__':
    H = get_hparams()
    vis = setup_visdom(H)
    if H.ae_load_step:
        config_log(H.log_dir)
        start_training_log(H.get_bert_param_dict())
        main(H)
    else:
        print(f'Please select an autoencoder to load using the --ae_load_step flag')

# TODO: try initialising from real latents instead.

