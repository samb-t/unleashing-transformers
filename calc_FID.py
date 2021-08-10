import torch_fidelity
from utils import *
from hparams import get_vqgan_hparams, get_sampler_hparams
from models import VQAutoEncoder, VQGAN, Generator
from tqdm import tqdm
from train_sampler import get_sampler
import torchvision
import imageio

'''
TODO: calculate FID for a given model, either VQGAN or Sampler

- [x] Load model Hparams
- [x] Load ae        images = (images * 255).clamp(0, 255).to(torch.uint8)
        images = TensorDataset(images)
- [x] Load VQGAN or Sampler
- [x] Load saved model
- [x] Generate + save samples
- [x] Run bash commands to calculate FID

'''

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

def main(H):
    vis = setup_visdom(H)
    
    if H.model == 'vqgan':
        model = VQGAN(H)
        data_loader = get_data_loader(H.dataset, H.img_size, H.batch_size, drop_last=False)
    else:
        quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
            H,
            ['quantize', 'generator'],
            remove_component_from_key=True
        )
        embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight')
        embedding_weight = embedding_weight.cuda()
        generator = Generator(H)
        generator.load_state_dict(quanitzer_and_generator_state_dict)
        # generator = generator.cuda()

        model = get_sampler(H, embedding_weight, None)

    if H.ema:
        model_name = H.model + '_ema'
    else:
        model_name = H.model

    model = load_model(model, model_name, H.load_step, H.load_dir).cuda()
    model = model.eval()
    model.n_samples = H.batch_size

    with torch.no_grad():
        if H.model == 'vqgan': 
            log('Generating VQGAN samples:')    
            idx = 0
            for x, *_ in tqdm(data_loader):
                x = x.cuda()
                x_hat, _, _ = model.ae(x)
                # vis.images(x_hat.clamp(0,1), win='recon_check', opts=dict(title='recon_check'))
                save_images(x_hat.detach().cpu(), f'recon', idx, f'{H.load_dir}_recons', save_indivudally=True)
                idx += 1
            
            images = BigDataset(f"logs/{H.load_dir}_recons/images/")
            model = model.cpu()
            model = None

        else:
            if H.dataset == 'cifar10':
                samples_needed = 50000
            elif H.dataset == 'churches':
                # Commonly use 50000 images of samples and training data - https://github.com/GaParmar/clean-fid
                # Gotta Go Fast only use 5k!
                samples_needed = 10000
            elif H.dataset == 'ffhq_256':
                ...

            if not H.n_samples:
                raise ValueError('Please specify number of samples to calculate per step using --n_samples')

            all_latents = []
            for _ in tqdm(range(int(samples_needed/H.batch_size) + 1)):
                latents = model.sample(num_steps=H.num_sample_steps)
                all_latents.append(latents.cpu())
            
            all_latents = torch.cat(all_latents, dim=0)
            torch.save(all_latents, 'all_latents_backup.pkl')
            embedding_weight = model.embedding_weight.cuda().clone()
            del model
            # model = None
            generator = generator.cuda()

            for idx, latents in tqdm(list(enumerate(torch.split(all_latents, H.vqgan_batch_size)))):
                latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size).cuda()
                q = torch.matmul(latents_one_hot, embedding_weight).view(
                    latents_one_hot.size(0), H.latent_shape[1], H.latent_shape[2], H.emb_dim
                ).permute(0, 3, 1, 2).contiguous()
                gen_images = generator(q)
                vis.images(gen_images[:64].clamp(0,1), win='sample_check', opts=dict(title='sample_check'))
                save_images(gen_images.detach().cpu(), f'sample', idx, f'{H.load_dir}_{H.num_sample_steps}_samples', save_indivudally=True)

            images = BigDataset(f"logs/{H.load_dir}_{H.num_sample_steps}_samples/images/")
            generator = generator.cpu()
            generator = None

        # samples_needed = 10000
        # images = BigDataset(f"logs/{H.load_dir}_samples/images/")
        # torch.save(images, f'images_{H.dataset}_backup')
        # images = (images * 255).clamp(0, 255).to(torch.uint8)
        # images = TensorDataset(images)
    

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
            input2_cache_name = f'lsun_churches'
        elif H.dataset == 'ffhq_256':
            ...
        
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
    if H.sampler is None:
        H.model = 'vqgan'
    else:
        H.model = H.sampler
        H.vqgan_batch_size = 32
        H.num_sample_steps = 99
    config_log(H.log_dir)
    log('---------------------------------')
    if H.load_step > 0:
        log(f'Calculating FID for {H.model} loaded from step {H.load_step}')  
        start_training_log(H)
        main(H) 
    else:
        raise ValueError('No value provided for load_step, cannot calculate FID for new model')


