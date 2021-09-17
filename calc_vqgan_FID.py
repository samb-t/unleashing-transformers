import torch_fidelity
import torch
from hparams import get_vqgan_hparams
from models import VQGAN, Generator
from tqdm import tqdm
from train_sampler import get_sampler
import torchvision
import imageio
import os
import copy
from utils.log_utils import log, display_images, setup_visdom, load_model, save_images, config_log, start_training_log
from utils.data_utils import get_data_loader
from utils.sampler_utils import get_samples, retrieve_autoencoder_components_state_dicts, latent_ids_to_onehot
from utils.vqgan_utils import unpack_vqgan_stats, load_vqgan_from_checkpoint, calc_FID


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    def __getitem__(self, index):
        return self.tensor[index].mul(255).clamp_(0, 255).to(torch.uint8)
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
    
    vqgan = VQGAN(H).cuda()
    ema_vqgan = copy.deepcopy(vqgan)
    optim = torch.optim.Adam(vqgan.ae.parameters(), lr=H.lr)
    d_optim = torch.optim.Adam(vqgan.disc.parameters(), lr=H.lr)

    if H.load_step > 0:
        _, _, _, ema_vqgan, _ = load_vqgan_from_checkpoint(H, vqgan, optim, d_optim, ema_vqgan)
        ema_vqgan = ema_vqgan.cuda()

    del vqgan
    del optim
    del d_optim

    _, val_loader = get_data_loader(
        H.dataset,
        H.img_size,
        H.batch_size,
        get_val_train_split=True
    )

    with torch.no_grad():

        images = []
        for i, x in enumerate(val_loader):
            x_hat, stats = ema_vqgan.val_iter(x[0].cuda(), H.load_step)
            images.append(x_hat)
            if i == 0:
                vis.images(x_hat.clamp(0,1), win='recon_sanity', opts=dict(title='recon_sanity'))

        

        del ema_vqgan

        images = torch.cat(images, dim=0).cpu()
        images = TensorDataset(images)
        
        if H.dataset == 'cifar10':
            input2 = 'cifar10-train'
            input2_cache_name = 'cifar10-train'
        elif H.dataset == 'churches':
            input2 = torchvision.datasets.LSUN('../../../data/LSUN', classes=['church_outdoor_val'], transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(256),
                torchvision.transforms.ToTensor()
            ]))
            # TODO: Maybe only compute stats for samples_needed images from the dataset?
            # Yes. SOTA on churches only uses 50k https://github.com/saic-mdal/CIPS/blob/main/calculate_fid.py
            # This is a good reference as it also uses torch fidelity
            input2 = NoClassDataset(input2)
            input2_cache_name = 'lsun_churches_val'
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
    H = get_vqgan_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    if H.load_step > 0:
        log(f'Calculating FID for {H.model} loaded from step {H.load_step}')  
        start_training_log(H)
        main(H, vis) 
    else:
        raise ValueError('No value provided for load_step, cannot calculate FID for new model')

