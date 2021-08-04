from utils import *
from hparams import get_training_hparams
from models import VQAutoEncoder, VQGAN
from tqdm import tqdm
from train_sampler import get_sampler

def main(H):
    ae = VQAutoEncoder(H).cuda()
    model = get_sampler(H, ae)

    if H.ema:
        model_name = H.model + '_ema'
    else:
        model_name = H.model

    model = load_model(model, model_name, H.load_step, H.load_dir).cuda()

    with torch.no_grad():
        if H.model == 'vqgan': 
            log('Generating VQGAN samples:')    
            recons = []   
            for x, *_ in tqdm(data_loader):
                x = x.cuda()
                x_hat, _, _ = model.ae(x)
                recons.append(x_hat.detach().cpu())

            images = torch.cat(recons, dim=0)

        else:
            if H.dataset == 'cifar10':
                samples_needed = 50000
            elif H.dataset == 'churches':
                ...
            elif H.dataset == 'ffhq_256':
                ...

            if not H.n_samples:
                raise ValueError('Please specify number of samples to calculate per step using --n_samples')
            
            samples = []
            for _ in range(int(samples_needed/H.n_samples) + 1):
                latents = model.sample()
                q = model.embed(latents)
                gen_images = ae.generator(q)
                samples.append(gen_images.detach().cpu())

            images = torch.cat(samples, dim=0)