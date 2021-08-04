import copy
import torch
import torch_fidelity
from tqdm import tqdm
from .data_utils import get_data_loader
from .log_utils import load_model, load_stats

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    def __getitem__(self, index):
        return self.tensor[index]
    def __len__(self):
        return self.tensor.size(0)


def load_vqgan_from_checkpoint(H, vqgan, optim, d_optim, ema_vqgan):
    vqgan = load_model(vqgan, 'vqgan', H.load_step, H.load_dir).cuda()
    if H.load_optim:
            optim = load_model(optim, 'ae_optim', H.load_step, H.load_dir)
            d_optim = load_model(d_optim, 'disc_optim', H.load_step, H.load_dir)

    if H.ema:
        try:
            ema_vqgan = load_model(
                            ema_vqgan,
                            f'vqgan_ema',
                            H.load_step, 
                            H.load_dir
                        )
        except:
            print('No ema model found')
            ema_vqgan = copy.deepcopy(vqgan)

    # return none if no associated saved stats
    try:
        train_stats = load_stats(H, H.load_step)
    except:
        train_stats = None
    return vqgan, optim, d_optim, ema_vqgan, train_stats


def unpack_vqgan_stats(stats):
    return (
        stats['losses'],
        stats['mean_losses'],
        stats['val_losses'],
        stats['latent_ids'],
        stats['fids'],
        stats['best_fid'],
        stats['steps_per_log'],
        stats['steps_per_eval']
    )


def calc_FID(H, model):
    images, recons = collect_ims_and_recons(H, model)
    images = TensorDataset(images)
    recons = TensorDataset(recons)
    fid = torch_fidelity.calculate_metrics(input1=recons, input2=images, 
        cuda=True, fid=True, verbose=True)["frechet_inception_distance"]
    return fid

@torch.no_grad()
def collect_ims_and_recons(H, model):
    
    data_iterator = get_data_loader(
        H.dataset,
        H.img_size,
        H.batch_size,
        drop_last=False,
        shuffle=False
    )

    images = []
    recons = []
    for x, *_ in tqdm(data_iterator):
        images.append(x)
        x = x.cuda()
        if H.deepspeed:
            x = x.half()
        x_hat, *_ = model.ae(x)
        recons.append(x_hat.detach().cpu())


    images = convert_to_RGB(torch.cat(images, dim=0))
    recons = convert_to_RGB(torch.cat(recons, dim=0))
    return images, recons


def convert_to_RGB(image_estimate):
    # images = torch.round((image_estimate * 255)).to(torch.uint8).clamp(0,255)
    images = (image_estimate * 255).clamp(0,255).to(torch.uint8)
    return images