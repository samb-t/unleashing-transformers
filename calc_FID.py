import torch_fidelity
from utils import *
from hparams import get_training_hparams
from models import VQAutoEncoder, VQGAN
from tqdm import tqdm
from train_sampler import get_sampler

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


def main(H):
    ae = VQAutoEncoder(H).cuda()
    
    if H.model == 'vqgan':
        H.batch_size = H.vqgan_batch_size
        model = VQGAN(ae, H)
        data_loader = get_data_loader(H.dataset, H.img_size, H.batch_size, drop_last=False)
    else:
        H.ae_load_step = H.load_step
        H.ae_load_dir = H.load_dir
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

        images = (images * 255).clamp(0, 255).to(torch.uint8)
        images = TensorDataset(images)

        if H.dataset == 'cifar10':
            input2 = 'cifar10-train'
        elif H.dataset == 'churches':
            ...
        elif H.dataset == 'ffhq_256':
            ...
        
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=images,
            input2=input2,
            cuda=True,
            fid=True,
            verbose=True
        )
        log(metrics_dict)


if __name__=='__main__':
    H = get_training_hparams()
    config_log(H.log_dir)
    log('---------------------------------')
    if H.load_step > 0:
        log(f'Calculating FID for {H.model} loaded from step {H.load_step}')  
        start_training_log(H)
        main(H) 
    else:
        raise ValueError('No value provided for load_step, cannot calculate FID for new model')


