import copy
import imageio
import os
import torch
import torch_fidelity
import torch.distributions as dists
import torchvision
from hparams import get_sampler_hparams
from models import VQGAN
from tqdm import tqdm
from train_sampler import get_sampler
from utils.log_utils import log, setup_visdom, load_model, save_images, config_log, start_training_log
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, latent_ids_to_onehot
from utils.vqgan_utils import load_vqgan_from_checkpoint


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)


class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_paths = os.listdir(folder)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = imageio.imread(self.folder+path)
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> channels first
        # How does torchvision save quantize?
        return img

    def __len__(self):
        return len(self.image_paths)


class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        self.length = length if length is not None else len(dataset)

    def __getitem__(self, index):
        return self.dataset[index][0].mul(255).clamp_(0, 255).to(torch.uint8)

    def __len__(self):
        return self.length


def main(H, vis):

    vqgan = VQGAN(H).cuda()
    ema_vqgan = copy.deepcopy(vqgan)
    optim = torch.optim.Adam(vqgan.ae.parameters(), lr=H.lr)
    d_optim = torch.optim.Adam(vqgan.disc.parameters(), lr=H.lr)

    if H.load_step > 0:
        _, _, _, ema_vqgan, _ = load_vqgan_from_checkpoint(
            H, vqgan,
            optim,
            d_optim,
            ema_vqgan,
            load_step=H.ae_load_step,
            load_dir=H.ae_load_dir
        )
        ema_vqgan = ema_vqgan.cuda()

    del vqgan
    del optim
    del d_optim

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight')
    embedding_weight = embedding_weight.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()

    if H.load_step > 0:
        sampler = load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir).cuda()

    sampler = sampler.eval()

    # Load images to be inpainted
    imgs = imageio.imread('inpainting_images.png')
    imgs = torch.from_numpy(imgs).permute(2, 0, 1)[:3].float() / 255  # ignore alpha channel
    masks = imageio.imread('inpainting_mask.png')
    masks = torch.from_numpy(masks).permute(2, 0, 1)[:3].float()/255

    # separate into individual images
    imgs = torch.stack([imgs[:, :, i*256:i*256+256] for i in range(imgs.size(-1)//256)], dim=0)
    masks = torch.stack([masks[:, :, i*256:i*256+256] for i in range(masks.size(-1)//256)], dim=0)

    # Convert 256x256 mask to 16x16 mask by max pooling
    masks = torch.nn.functional.max_pool2d(1-masks, 16, stride=16)
    masks = masks.mean(dim=1) > 1e-4 # epsilon
    masks = masks.reshape(masks.size(0), -1).cuda()

    device = 'cuda'

    # Check VQGAN reconstruction
    with torch.no_grad():
        x_hat, stats = ema_vqgan.val_iter(imgs.cuda(), H.ae_load_step)
        vis.images(x_hat.clamp(0, 1), win='recon_check')

        latents = stats['latent_ids']
        latents[masks] = H.codebook_size

        unmasked_latents = []

        for latent, mask in zip(latents, masks):
            x_t, mask = latent.unsqueeze(0), mask.unsqueeze(0)
            num_mask = torch.sum(mask.float() > 0)
            start_time_step = num_mask + 1
            print(f"Starting at time step {start_time_step}")

            unmasked = torch.bitwise_not(mask)

            for t in reversed(range(1, start_time_step)):
                print(f'Sample timestep {t:4d}', end='\r')
                t = torch.full((1,), t, device=device, dtype=torch.long)

                # where to unmask
                changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)
                # don't unmask somewhere already unmasked
                changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
                # update mask with changes
                unmasked = torch.bitwise_or(unmasked, changes)

                # x_t, _, _ = self.q_sample(x_0, t)
                x_0_logits = sampler._denoise_fn(x_t, t=t)
                # if self.mask is not None:
                #     x_0_logits = x_0_logits + self.mask.reshape(1,1,-1)
                # scale by temperature
                x_0_logits = x_0_logits / H.temp
                x_0_dist = dists.Categorical(
                    logits=x_0_logits)
                x_0_hat = x_0_dist.sample().long()
                x_t[changes] = x_0_hat[changes]

            unmasked_latents.append(x_t)

        unmasked_latents = torch.cat(unmasked_latents, dim=0)
        latents_one_hot = latent_ids_to_onehot(unmasked_latents, H.latent_shape, H.codebook_size).cuda()
        q = torch.matmul(latents_one_hot, embedding_weight).view(
            latents_one_hot.size(0), H.latent_shape[1], H.latent_shape[2], H.emb_dim
        ).permute(0, 3, 1, 2).contiguous()

        gen_images = ema_vqgan.ae.generator(q)

        vis.images(gen_images.clamp(0,1), win='gen_images')


    # exit()

    with torch.no_grad():

        # print(f'Sampling with temperature {H.temp}')
        # all_latents = []
        # for i in tqdm(range(int(samples_needed/H.batch_size) + 1)):
        #     if H.sample_type == 'default':
        #         latents = sampler.sample(temp=H.temp, sample_stride=sample_stride, sample_steps=sample_steps)
        #     elif H.sample_type == 'v2':
        #         latents = sampler.sample_v2(temp=H.temp, sample_stride=sample_stride, sample_steps=sample_steps)
        #     torch.save(latents.cpu(), f"logs/{image_dir}/latents_backup_{i}.pkl")
        #     all_latents.append(latents.cpu())

        # # all_latents = [torch.load(f"logs/{image_dir}/latents_backup_{i}.pkl") for i in range(46)] + [torch.load(f"logs/{image_dir}/part2_latents_backup_{i}.pkl") for i in range(48)] + [torch.load(f"logs/{image_dir}/part3_latents_backup_{i}.pkl") for i in range(9)]
        # all_latents = torch.cat(all_latents, dim=0)
        # torch.save(all_latents, f"logs/{image_dir}/all_latents_backup.pkl")
        all_latents = torch.load(f"logs/{image_dir}/all_latents_backup.pkl")
        embedding_weight = sampler.embedding_weight.cuda().clone()
        # sampler = sampler.cpu()
        del sampler

        all_latents = all_latents.cuda()



        for idx, latents in tqdm(list(enumerate(torch.split(all_latents, H.vqgan_batch_size)))):
            latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size).cuda()
            q = torch.matmul(latents_one_hot, embedding_weight).view(
                latents_one_hot.size(0), H.latent_shape[1], H.latent_shape[2], H.emb_dim
            ).permute(0, 3, 1, 2).contiguous()
            gen_images = generator(q)
            # vis.images(gen_images[:64].clamp(0,1), win='FID_sample_check', opts=dict(title='FID_sample_check'))
            save_images(gen_images.detach().cpu(), f'sample', idx, image_dir, save_indivudally=False)#, save_indivudally=True)
            images = BigDataset(f"logs/{image_dir}/images/")
        # generator = generator.cpu()
        del generator

        images = BigDataset(f"logs/{image_dir}/images/")


        if H.dataset == 'cifar10':
            input2 = 'cifar10-train'
            input2_cache_name = 'cifar10-train'
        elif H.dataset == 'churches':
            input2 = torchvision.datasets.LSUN('../../../data/LSUN', classes=['church_outdoor_train'], transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(256),
                torchvision.transforms.ToTensor()
            ]))
            # TODO: Maybe only compute stats for samples_needed images from the dataset?
            # Yes. SOTA on churches only uses 50k https://github.com/saic-mdal/CIPS/blob/main/calculate_fid.py
            # This is a good reference as it also uses torch fidelity
            input2 = NoClassDataset(input2)
            input2_cache_name = 'lsun_churches'
        elif H.dataset == 'bedrooms':
            input2 = torchvision.datasets.LSUN('/projects/cgw/lsun', classes=['bedroom_train'], transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(256),
                torchvision.transforms.ToTensor()
            ]))
            # TODO: Maybe only compute stats for samples_needed images from the dataset?
            # Yes. SOTA on churches only uses 50k https://github.com/saic-mdal/CIPS/blob/main/calculate_fid.py
            # This is a good reference as it also uses torch fidelity
            input2 = NoClassDataset(input2, length=50000)
            input2_cache_name = 'lsun_bedroom_train_50k'
        elif H.dataset == 'ffhq':
            input2 = torchvision.datasets.ImageFolder('~/Repos/_datasets/FFHQ',  transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor()
            ]))
            input2 = NoClassDataset(input2)
            input2_cache_name = 'ffhq'

        metrics_dict = torch_fidelity.calculate_metrics(
            input1=images,
            input2=input2,
            cuda=True,
            fid=True,
            verbose=True,
            input2_cache_name=input2_cache_name
        )
        log(metrics_dict)


if __name__=='__main__':
    H = get_sampler_hparams()
    H.vqgan_batch_size = 32
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    if H.load_step > 0:
        log(f'Calculating FID for {H.model} loaded from step {H.load_step}')
        start_training_log(H)
        main(H, vis)
    else:
        raise ValueError('No value provided for load_step, cannot calculate FID for new model')