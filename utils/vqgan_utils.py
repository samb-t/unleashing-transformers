import copy
import torch
import torch_fidelity
from tqdm import tqdm
from .data_utils import get_data_loaders
from .log_utils import load_model, load_stats, log


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
            ema_vqgan = load_model(
                            ema_vqgan,
                            "vqgan_ema",
                            H.load_step,
                            H.load_dir
                        )
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


# TODO: replace with non-manual key specification
def unpack_vqgan_stats(stats):
    return (
        stats["losses"],
        stats["mean_losses"],
        stats["val_losses"],
        stats["latent_ids"],
        stats["fids"],
        stats["best_fid"],
        stats["steps_per_log"],
        stats["steps_per_eval"]
    )


def calc_FID(H, model):
    images, recons = collect_ims_and_recons(H, model)
    images = TensorDataset(images)
    recons = TensorDataset(recons)
    fid = torch_fidelity.calculate_metrics(
        input1=recons,
        input2=images,
        cuda=True,
        fid=True,
        verbose=True,
    )["frechet_inception_distance"]

    return fid


@torch.no_grad()
def collect_ims_and_recons(H, model):
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
    images = []
    recons = []
    for x, *_ in tqdm(iter(data_loader)):
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
    images = (image_estimate * 255).clamp(0, 255).to(torch.uint8)
    return images
