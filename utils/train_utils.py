import torch
import torchvision
import os
from torchvision import transforms
from torchvision.transforms.transforms import CenterCrop
import visdom
from tqdm import tqdm
from .log_utils import log, save_latents
from .model_utils import MyOneHotCategorical, BERTDataset


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def cycle(iterable, encode_to_one_hot=False, H=None):
    while True:
        for x in iterable:
            # if processing latents, encode to one hots and wrap in list (now works with vqgan data)
            if encode_to_one_hot:
                yield [latent_ids_to_onehot(x, H.latent_shape, H.codebook_size)]
            else:
                yield x


def setup_visdom(H):
    if H.ncc:
        server = 'ncc1.clients.dur.ac.uk'
    else:
        server = None

    if server:
        vis = visdom.Visdom(server=server, port=H.visdom_port)
    else:
        vis = visdom.Visdom(port=H.visdom_port)
    return vis


def generate_latent_ids(H, ae, dataloader):
    latent_ids = []
    for x, _ in tqdm(dataloader):
        x = x.cuda()
        z = ae.encoder(x) # B, emb_dim, H, W

        z = z.permute(0, 2, 3, 1).contiguous() # B, H, W, emb_dim
        z_flattened = z.view(-1, H.emb_dim) # B*H*W, emb_dim

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + \
            (ae.quantize.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, ae.quantize.embedding.weight.t())
        
        min_encoding_indices = torch.argmin(d, dim=1)

        latent_ids.append(min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous())

    latent_ids_out = torch.cat(latent_ids, dim=0)

    return latent_ids_out


def latent_ids_to_onehot(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
    encodings = torch.zeros(
        min_encoding_indices.shape[0],
        codebook_size
    ).to(latent_ids.device)
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(
        latent_ids.shape[0],
        latent_shape[1],
        latent_shape[2],
        codebook_size
    )
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)


def get_init_dist(H, latent_loader, cuda=False):
    init_dist_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_init_dist'
    if os.path.exists(init_dist_filepath):
        log(f'Loading init distribution from {init_dist_filepath}')
        init_dist = torch.load(init_dist_filepath)
    else:  
        # latents # B, H*W, codebook_size
        eps = 1e-3 / H.codebook_size

        batch_sum = torch.zeros(H.latent_shape[1]*H.latent_shape[2], H.codebook_size).cuda()
        log('Generating init distribution:')
        for batch in tqdm(latent_loader):
            batch = batch.cuda()
            latents = latent_ids_to_onehot(batch, H.latent_shape, H.codebook_size)
            batch_sum += latents.sum(0)

        init_mean = batch_sum / (len(latent_loader) * H.batch_size)

        init_mean += eps # H*W, codebook_size
        init_mean = init_mean / init_mean.sum(-1)[:, None] # renormalize pdfs after adding eps
        if cuda:
            init_dist = MyOneHotCategorical(init_mean.cuda())
        else:
            init_dist = MyOneHotCategorical(init_mean.cpu())
        
        torch.save(init_dist, f'latents/{H.dataset}_{H.latent_shape[-1]}_init_dist')

    return init_dist

def get_data_loader(dataset_name, img_size, batch_size, num_workers=1, train=True, drop_last=True, download=False, shuffle=True):
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST('~/Repos/_datasets', train=train, download=download, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ]))
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('~/Repos/_datasets/CIFAR10', train=train, download=download, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ]))
    elif dataset_name == 'flowers':
        dataset = torchvision.datasets.ImageFolder('~/Repos/_datasets/flowers', transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor()
        ]))
    elif dataset_name == 'churches':
        dataset = torchvision.datasets.LSUN('../Repos/_datasets/LSUN', classes=['church_outdoor_train'], transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor()
        ]))
    elif dataset_name =='celeba':
        dataset = torchvision.datasets.CelebA('~/Repos/_datasets/celeba', download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor()
        ]))
    elif dataset_name == 'ffhq':
        dataset = torchvision.datasets.ImageFolder('~/Repos/_datasets/FFHQ',  transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ]))
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, sampler=None, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
    return loader


# TODO: only use iterator or loader not both, need to rewrite code for init_dist generation
def get_latent_loaders(H, ae):
    latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_latents'
    if os.path.exists(latents_filepath):
        latent_ids = torch.load(latents_filepath)
    else:
        full_dataloader = get_data_loader(H.dataset, H.img_size, H.vqgan_batch_size, drop_last=False, shuffle=False)
        ae = ae.cuda() # put ae on GPU for generating
        latent_ids = generate_latent_ids(H, ae, full_dataloader)
        ae = ae.cpu() # put back onto CPU to save memory during EBM training
        save_latents(latent_ids, H.dataset, H.latent_shape[-1])

    latent_loader = torch.utils.data.DataLoader(latent_ids, batch_size=H.batch_size, shuffle=True)
    
    # if using masked dataset (might need to add more conditionals here)
    if H.model == 'bert': 
        masked_latent_ids = BERTDataset(latent_ids, H.codebook_size, H.codebook_size)
        latent_iterator = cycle(torch.utils.data.DataLoader(
            masked_latent_ids, 
            batch_size=H.batch_size, 
            shuffle=False
        ))
    else:
        latent_iterator = cycle(latent_loader, encode_to_one_hot=(H.model=='ebm'), H=H)
    
    return latent_loader, latent_iterator


@torch.no_grad()
def collect_recons(H, model, data_iterator):
    recons = []
    for x, *_ in data_iterator:
        x = x.cuda()
        x_hat, *_ = model.ae(x)
        recons.append(x_hat.detach().cpu())
    return torch.cat(recons, dim=0)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    def __getitem__(self, index):
        return self.tensor[index]
    def __len__(self):
        return self.tensor.size(0)
