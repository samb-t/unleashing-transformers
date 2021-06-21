import torch
import visdom
import time
from vqgan import *
from unet import Unet
from bert import BERTDataset, warm_start_from_real
from energy import *
from hparams import get_hparams
from utils import *

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
    )
    ae = load_model(ae, 'ae', H.ae_load_step, f'vqgan_test_{H.dataset}')

    if os.path.exists(f'latents/{H.dataset}_latents_{H.ae_load_step}'):
        latent_ids = torch.load(f'latents/{H.dataset}_latents_{H.ae_load_step}')
    else:
        full_dataloader = get_data_loader(H.dataset, H.img_size, 4, drop_last=False, shuffle=False)

        latent_ids = generate_latent_ids(ae, full_dataloader, H)
        save_latents(latent_ids, H.dataset, H.ae_load_step)
    
    print("latent min", latent_ids.min(), "latent max", latent_ids.max())
    print(torch.unique(latent_ids).shape)

    unmasked_latents = latent_ids[:32].clone()
    
    latent_dataset = BERTDataset(latent_ids.reshape(latent_ids.size(0), -1), H.codebook_size, H.codebook_size)
    latent_loader = torch.utils.data.DataLoader(latent_dataset, batch_size=H.ebm_batch_size, shuffle=False)
    latent_iterator = cycle(latent_loader)

    # Simple Transformer
    unet = Unet(64)
    optim = torch.optim.Adam(unet.parameters(), lr=1e-4)

    start_step = 0
    if H.load_step > 0:
        transformer = load_model(unet, 'unet', H.load_step, log_dir)
        if H.load_optim:
            optim = load_model(optim, 'unet_optim', H.load_step, log_dir)
        start_step = H.load_step

    log(f'Transformer Parameters: {len(ptv(transformer.parameters()))}')

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

        # TODO: Print accuracy
        accuracy = (logits.max(-1)[1][target > 0] == target[target > 0]).float().mean()

        step_time_finish = time.time()
        step_time_taken = step_time_finish - step_time_start

        # display + save less frequently than ebm
        if step % 1 == 0 and step > 0:
            log(f"Step: {step}, time: {step_time_taken:.3f}, nll={nll:.4f}, grad_norm={grad_norm:.4f}, accuracy={accuracy:.4f}")

        if step % 100 == 0:
            # probs = F.softmax(logits, dim=-1)
            # ix = torch.multinomial(probs, num_samples=1)
            logits = logits.max(2)[1]
            q = transformer.embed(latent_ids_to_onehot(logits.reshape(-1, 16, 16).contiguous(), H))
            with torch.no_grad():
                samples = ae.generator(q.cpu())
            vis.images(samples[:64].clamp(0,1), win='recons', opts=dict(title='recons'))

        if step % 1000 == 0 and step > 0:
            # samples = warm_start(transformer, H.codebook_size)

            # print("Here 1", latent_ids.max())
            samples = warm_start_from_real(unmasked_latents.cuda(), transformer, H.codebook_size)

            # print(samples.max(), samples.min())
            # print(samples)

            q = transformer.embed(latent_ids_to_onehot(samples.reshape(-1, 16, 16).contiguous(), H))
            with torch.no_grad():
                samples = ae.generator(q.cpu())

            vis.images(samples[:64].clamp(0,1), win='samples', opts=dict(title='samples'))

if __name__ == '__main__':
    H = get_hparams()
    # vis = visdom.Visdom(server='http://ncc1.clients.dur.ac.uk', port=H.vis_port)
    vis = visdom.Visdom()
    if H.ae_load_step:
        log_dir = f'bert_{H.dataset}'
        config_log(log_dir)
        start_training_log(H.get_ebm_param_dict())
        main(H)
    else:
        print(f'Please select an autoencoder to load using the --ae_load_step flag')