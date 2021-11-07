import imageio
import os
import torch
import torch_fidelity
import torchvision
from hparams import get_sampler_hparams
from models import Generator
from utils.log_utils import log, load_model, save_images, config_log, start_training_log
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, latent_ids_to_onehot
from utils.experiment_utils import generate_samples
from tqdm import tqdm
from train_sampler import get_sampler


class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_paths = os.listdir(folder)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = imageio.imread(self.folder+path)
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> channels first
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

    images = generate_samples(H, ...)

    if H.dataset == "cifar10":
        input2 = "cifar10-train"
        input2_cache_name = "cifar10-train"
    elif H.dataset == "churches":
        input2 = torchvision.datasets.LSUN(
            "projects/cgw/LSUN",
            classes=["church_outdoor_train"],
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(256),
                torchvision.transforms.ToTensor()
            ])
        )
        input2 = NoClassDataset(input2)
        input2_cache_name = "lsun_churches"
    elif H.dataset == "ffhq":
        input2 = torchvision.datasets.ImageFolder(
            "/projects/cgw/FFHQ",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor()
            ])
        )
        input2 = NoClassDataset(input2)
        input2_cache_name = "ffhq"

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=images,
        input2=input2,
        cuda=True,
        fid=True,
        verbose=True,
        input2_cache_name=input2_cache_name
    )
    log(metrics_dict)


if __name__ == "__main__":
    H = get_sampler_hparams(get_FID_args=True)
    H.vqgan_batch_size = 32
    if H.log_dir == "test":  # i.e. if it hasn"t been set using a flag)
        H.log_dir = f"{H.load_dir}_FID_samples"
    config_log(H.log_dir)
    log("---------------------------------")
    if H.load_step > 0:
        log(f"Calculating FID for {H.model} loaded from step {H.load_step}")
        start_training_log(H)
        main(H)
    else:
        raise ValueError("No value provided for load_step, cannot calculate FID for new model")
