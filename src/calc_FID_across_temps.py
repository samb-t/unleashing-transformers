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


def load_models(H):
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

    log(f'Number of sampler parameters: {len(torch.nn.utils.parameters_to_vector(sampler.parameters()))}')

    sampler = sampler.eval()

    return generator, sampler

def main(H, vis):
    
    generator, sampler = load_models(H)

    start_temp = 80
    end_temp = 89

    if H.stepping is not None:
        sample_stride, sample_steps = H.stepping.split('-')
        if sample_stride == 'magic':
            sample_steps = int(sample_steps)

    with torch.no_grad():
        for temp_int in range(start_temp, end_temp+1):
            sampler = sampler.cuda()
            temp = temp_int / 100
            log(f'Sampling {H.n_samples} images with temperature {temp}')
            all_latents = []
            for _ in tqdm(range(int(H.n_samples/H.batch_size) + 1)):
                if H.sampler == 'absorbing':
                    if H.sample_type == 'default':
                        latents = sampler.sample(temp=temp, sample_stride=sample_stride, sample_steps=sample_steps)
                    elif H.sample_type == 'v2':
                        latents = sampler.sample_v2(temp=temp, sample_stride=sample_stride, sample_steps=sample_steps)
                else:
                    latents = sampler.sample(temp=H.temp)
                all_latents.append(latents.cpu())

            all_latents = torch.cat(all_latents, dim=0)
            # torch.save(all_latents, f')
              
            im_dir = f'{H.log_dir}/{str(temp).replace(".", "")}'

            ######### COMMENT THIS OUT WHEN NOT USING PREGENERATED LATENTS #################

            # all_latents = torch.load(f'latents_{H.log_dir}_step_{H.load_step}_n_{H.n_samples}_temp_{str(temp).replace(".", "")}_backup.pkl')

            embedding_weight = sampler.embedding_weight.cuda().clone()
            sampler = sampler.cpu()
            # del sampler

            generator = generator.cuda()
            for idx, latents in tqdm(list(enumerate(torch.split(all_latents, H.vqgan_batch_size)))):
                latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size).cuda()
                q = torch.matmul(latents_one_hot, embedding_weight).view(
                    latents_one_hot.size(0), H.latent_shape[1], H.latent_shape[2], H.emb_dim
                ).permute(0, 3, 1, 2).contiguous()
                gen_images = generator(q)
                vis.images(gen_images[:64].clamp(0,1), win='FID_sample_check', opts=dict(title='FID_sample_check'))
                save_images(gen_images.detach().cpu(), f'sample', idx, im_dir, save_individually=True)
            
            images = BigDataset(f"logs/{im_dir}/images/")
            generator = generator.cpu()
            # generator = None        

            if H.dataset == 'cifar10':
                input2 = 'cifar10-train'
                input2_cache_name = 'cifar10-train'
            elif H.dataset == 'churches':
                input2 = torchvision.datasets.LSUN('../../../data/LSUN', classes=['church_outdoor_train'], transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(256),
                    torchvision.transforms.ToTensor()
                ]))

                input2 = NoClassDataset(input2)
                input2_cache_name = 'lsun_churches'
            elif H.dataset == 'ffhq':
                input2 = torchvision.datasets.ImageFolder('/projects/cgw/FFHQ',  transform=torchvision.transforms.Compose([
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
    if H.log_dir == 'test': #i.e. if it hasn't been set using a flag)
        H.log_dir = f"{H.load_dir}_FID_multitemp_samples"
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    if H.load_step > 0:
        log(f'Calculating FID for {H.sampler} loaded from step {H.load_step}')  
        start_training_log(H)
        main(H, vis) 
    else:
        raise ValueError('No value provided for load_step, cannot calculate FID for new model')

