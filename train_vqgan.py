import torch
import lpips
import visdom
from vqgan import *
from hparams import get_hparams
from utils import *


# %% main training loop
def main(H): 
    train_iterator = cycle(get_data_loader(H.dataset, H.img_size, H.vqgan_batch_size, num_workers=8))
    
    autoencoder = VQAutoEncoder(
        H.n_channels, 
        H.nf, 
        H.res_blocks, 
        H.codebook_size, 
        H.emb_dim, 
        H.ch_mult, 
        H.img_size, 
        H.attn_resolutions).cuda()
    
    discriminator = Discriminator(
        H.n_channels, 
        H.ndf, 
        n_layers=H.disc_layers).cuda()
    perceptual_loss = lpips.LPIPS(net='vgg').cuda()

    ae_optim = torch.optim.Adam(autoencoder.parameters(), lr=H.vqgan_lr, betas=(0.5,0.9))
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=H.vqgan_lr, betas=(0.5,0.9))

    start_step = 0 
    # load previous model from checkpoint
    if H.load_step > 0:
        autoencoder = load_model(autoencoder, 'ae', H.load_step, log_dir)
        discriminator = load_model(discriminator, 'discriminator', H.load_step, log_dir)
        ae_optim = load_model(ae_optim, 'ae_optim', H.load_step, log_dir)
        d_optim = load_model(d_optim, 'disc_optim', H.load_step, log_dir)
        start_step = H.load_step

    log(f'AE Parameters: {len(ptv(autoencoder.parameters()))}')
    log(f'Discriminator Parameters: {len(ptv(discriminator.parameters()))}')

    g_losses, d_losses = np.array([]), np.array([])

    for step in range(start_step, H.train_steps):
        x, _ = next(train_iterator)
        x = x.cuda()
        x_hat, codebook_loss = autoencoder(x)
        
        # get recon/perceptual loss
        recon_loss = torch.abs(x.contiguous() - x_hat.contiguous()) # L1 loss
        p_loss = perceptual_loss(x.contiguous(), x_hat.contiguous())
        nll_loss = recon_loss + H.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss)

        # update generator on every training step
        logits_fake = discriminator(x_hat.contiguous())
        g_loss = -torch.mean(logits_fake)
        last_layer = autoencoder.generator.blocks[-1].weight
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer)
        d_weight *= adopt_weight(1, step, H.disc_start_step)
        loss = nll_loss + d_weight * g_loss + codebook_loss
        g_losses = np.append(g_losses, loss.item())

        ae_optim.zero_grad()
        loss.backward()
        ae_optim.step()

        # update discriminator
        if step > H.disc_start_step:
            logits_real = discriminator(x.contiguous().detach()) # detach so that generator isn't also updated
            logits_fake = discriminator(x_hat.contiguous().detach())
            d_loss = hinge_d_loss(logits_real, logits_fake)
            d_losses = np.append(d_losses, d_loss.item())

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

        if step % H.steps_per_log == 0:
            if len(d_losses) == 0:
                d_loss_str = 'N/A'
            else:
                d_loss_str = f'{d_losses.mean():.3f}'
            
            log(f"Step {step}  G Loss: {g_losses.mean():.3f}  D Loss: {d_loss_str}  L1: {recon_loss.mean().item():.3f}  Perceptual: {p_loss.mean().item():.3f}  Disc: {g_loss.item():.3f}")
            g_losses, d_losses = np.array([]), np.array([])
        
        if step % H.steps_per_display_recons == 0:
            vis.images(x.clamp(0,1)[:64], win="x", nrow=int(np.sqrt(H.vqgan_batch_size)), opts=dict(title="x"))
            vis.images(x_hat.clamp(0,1)[:64], win="recons", nrow=int(np.sqrt(H.vqgan_batch_size)), opts=dict(title="recons"))
            
        if step % H.steps_per_save_recons == 0:
            save_images(x_hat[:64], vis, 'recons', step, log_dir)

        if step % H.steps_per_vqgan_checkpoint == 0 and step > 0 and not (step == H.load_step):
            print("Saving model")
            save_model(autoencoder, 'ae', step, log_dir)
            save_model(discriminator, 'discriminator', step, log_dir)
            save_model(ae_optim, 'ae_optim', step, log_dir)
            save_model(d_optim, 'disc_optim', step, log_dir)


if __name__ == '__main__':
    vis = visdom.Visdom()
    H = get_hparams()
    log_dir = f'vqgan_test_{H.dataset}'
    config_log(log_dir)
    start_training_log(H.get_vqgan_param_dict())
    main(H)