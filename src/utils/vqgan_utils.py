import copy
import torch
import torch_fidelity
import torch.nn.functional as F
from tqdm import tqdm
from .data_utils import get_data_loaders, BigDataset, NoClassDataset, get_datasets
from .log_utils import load_model, load_stats, log, save_images


def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


@torch.jit.script
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def calculate_adaptive_weight(recon_loss, g_loss, last_layer, disc_weight_max):
    recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
    return d_weight


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)


# TODO: replace this with general checkpointing method
def load_vqgan_from_checkpoint(H, vqgan, optim, disc_optim, ema_vqgan):
    vqgan = load_model(vqgan, "vqgan", H.load_step, H.load_dir).cuda()
    if H.load_optim:
        optim = load_model(optim, "ae_optim", H.load_step, H.load_dir)
        disc_optim = load_model(disc_optim, "disc_optim", H.load_step, H.load_dir)

    if H.ema:
        try:
            ema_vqgan = load_model(ema_vqgan, "vqgan_ema", H.load_step, H.load_dir)
        except FileNotFoundError:
            log("No EMA model found, starting EMA from model load point", level="warning")
            ema_vqgan = copy.deepcopy(vqgan)

    # return none if no associated saved stats
    try:
        train_stats = load_stats(H, H.load_step)
    except FileNotFoundError:
        log("No stats file found - starting stats from load step.")
        train_stats = None
    return vqgan, optim, disc_optim, ema_vqgan, train_stats


def calc_FID(H, model):
    generate_recons(H, model)
    real_dataset, _ = get_datasets(H.dataset, H.img_size, custom_dataset_path=H.custom_dataset_path)
    real_dataset = NoClassDataset(real_dataset)
    recons = BigDataset(f"logs/{H.log_dir}/FID_recons/images/")
    fid = torch_fidelity.calculate_metrics(
        input1=real_dataset,
        input2=recons,
        cuda=True,
        fid=True,
        verbose=True,
        input2_cache_name=f"{H.dataset}_recon_cache" if H.dataset != "custom" else None,
    )["frechet_inception_distance"]

    return fid


@torch.no_grad()
def generate_recons(H, model):
    # if using validation on FFHQ, don't want to include validation set images in FID calc
    training_with_validation = True if H.steps_per_eval else False

    data_loader, _ = get_data_loaders(
        H.dataset,
        H.img_size,
        H.batch_size,
        get_val_dataloader=training_with_validation,
        drop_last=False,
        shuffle=False,
    )
    log("Generating recons for FID calculation")

    for idx, x in tqdm(enumerate(iter(data_loader))):
        x = x[0].cuda()  # TODO check this for multiple datasets
        x_hat, *_ = model.ae(x)
        save_images(x_hat, "recon", idx, f"{H.log_dir}/FID_recons", save_individually=True)
        break
