import torch_fidelity
import torch
import torch.nn as nn
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
from utils.vqgan_utils import unpack_vqgan_stats, load_vqgan_from_checkpoint, calc_FID
import copy
import math

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
    vqgan.ae.generator.logsigma = nn.Sequential(
                                        nn.Conv2d(vqgan.ae.generator.final_block_ch, vqgan.ae.generator.final_block_ch, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(vqgan.ae.generator.final_block_ch, H.n_channels, kernel_size=1, stride=1, padding=0)).cuda()
    vqgan = load_model(vqgan, 'vqgan', 0, H.ae_load_dir).cuda()

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight')
    embedding_weight = embedding_weight.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()
    if H.load_step > 0:
        sampler =  load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir).cuda()
    
    sampler = sampler.eval()
    sampler.num_timesteps = 256

    _, val_loader = get_data_loader(
        H.dataset,
        H.img_size,
        H.batch_size,
        get_val_train_split=True,
        drop_last=False
    )

    with torch.no_grad():

        bpds = []
        for step, x in tqdm(enumerate(val_loader), total=len(val_loader)):
            if isinstance(x, list):
                x = x[0]
            x = x.cuda()
                        
            x_hat, stats = vqgan.probabilistic(x, step)
            nl_p_x_z = stats['nll_raw']

            z = stats['latent_ids']
            nl_p_z = sampler.elbo(z)

            pixels = 256 * 256 * 3
            # add and adjust for scaling (Baseed on GLOW. Not sure why 256^2 and not 256^2*3)
            # However, VDVAE uses 256^2*3 so...
            # Probably easier to just scale variable from 0-1 to 0-255
            # I.e. multiply mean by 256 and variance by 256^2 (add log(256) to logsigma)
            nl_p_x = nl_p_x_z + nl_p_z + float(math.log(32.) * pixels) # 5 bit
            bpd = nl_p_x / (pixels * math.log(2.0))
            bpds.extend(bpd.tolist())
            print(bpd.mean(), nl_p_x_z.mean(), nl_p_z.mean())
            print("ELBO So far", torch.tensor(bpds).mean())
        
        print("ELBO", torch.tensor(bpds).mean())




if __name__=='__main__':
    H = get_sampler_hparams()
    H.vqgan_batch_size = 32
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Calculating ELBO for {H.model}')  
    start_training_log(H)
    main(H, vis) 

