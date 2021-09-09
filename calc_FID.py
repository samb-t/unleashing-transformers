import torch_fidelity
import torch
from hparams import get_sampler_hparams
from models import VQGAN, Generator
from tqdm import tqdm
from train_sampler import get_sampler
import torchvision
import imageio
import os
from utils.log_utils import log, display_images, setup_visdom, load_model, save_images, config_log, start_training_log
from utils.data_utils import get_data_loader
from utils.sampler_utils import get_samples, retrieve_autoencoder_components_state_dicts, latent_ids_to_onehot

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
        img = torch.from_numpy(img).permute(2,0,1) # -> channels first
        # How does torchvision save quantize?
        return img

    def __len__(self):
        return len(self.image_paths)


class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        return self.dataset[index][0].mul(255).clamp_(0, 255).to(torch.uint8)
    
    def __len__(self):
        return len(self.dataset)

def main(H, vis):
    
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight')
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)

    #NOTE: can move generator to cpu to save memory if needbe - add flag?
    generator.load_state_dict(quanitzer_and_generator_state_dict)
    sampler = get_sampler(H, embedding_weight).cuda()

    if H.load_step > 0:
        sampler =  load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir).cuda()

    sampler = sampler.eval()

    with torch.no_grad():

        image_dir = H.FID_images_dir if H.FID_images_dir is not None else f"{H.load_dir}_samples"
        
        if H.samples_needed is not None:
            samples_needed = H.samples_needed
        elif H.dataset == 'cifar10':
            samples_needed = 50000

        elif H.dataset == 'churches':
            samples_needed = 2500

        elif H.dataset == 'ffhq':
            samples_needed = 10000

        if not H.n_samples:
            raise ValueError('Please specify number of samples to calculate per step using --n_samples')
        
        print(f'Sampling with temperature {H.temp}')
        all_latents = []
        for i in tqdm(range(int(samples_needed/H.batch_size) + 1)):
            latents = sampler.sample(temp=H.temp)
            torch.save(latents.cpu(), f"logs/{image_dir}/latents_backup_{i}.pkl")
            all_latents.append(latents.cpu())

        all_latents = torch.cat(all_latents, dim=0)
        torch.save(all_latents, f"logs/{image_dir}/all_latents_backup.pkl")
        # all_latents = torch.load(f"logs/{image_dir}/images/all_latents_backup.pkl")
        embedding_weight = sampler.embedding_weight.cuda().clone()
        # sampler = sampler.cpu()
        del sampler

        all_latents = all_latents.cuda()
        generator = generator.cuda()

        

        for idx, latents in tqdm(list(enumerate(torch.split(all_latents, H.vqgan_batch_size)))):
            latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size).cuda()
            q = torch.matmul(latents_one_hot, embedding_weight).view(
                latents_one_hot.size(0), H.latent_shape[1], H.latent_shape[2], H.emb_dim
            ).permute(0, 3, 1, 2).contiguous()
            gen_images = generator(q)
            # vis.images(gen_images[:64].clamp(0,1), win='FID_sample_check', opts=dict(title='FID_sample_check'))
            save_images(gen_images.detach().cpu(), f'sample', idx, image_dir, save_indivudally=True)
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

