import torch
from tqdm import tqdm
from .
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
    images = torch.round((image_estimate * 255)).to(torch.uint8).clamp(0,255)
    return images