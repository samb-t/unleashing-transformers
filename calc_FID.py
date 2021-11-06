import imageio
import os
import torch
import torch_fidelity
import torchvision
from hparams import get_sampler_hparams
from models import Generator
from tqdm import tqdm
from train_sampler import get_sampler
from utils.log_utils import log, load_model, save_images, config_log, start_training_log
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, latent_ids_to_onehot


class BigDataset(torch.utils.data.Dataset):

    def __init__(self, folder):
        self.folder = folder
        self.image_paths = os.listdir(folder)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = imageio.imread(self.folder+path)
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> channels first
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


def main(H):

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ["quantize", "generator"],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop("embedding.weight")
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)

    # NOTE: can move generator to cpu to save memory if needbe - add flag?
    generator.load_state_dict(quanitzer_and_generator_state_dict)
    sampler = get_sampler(H, embedding_weight).cuda()

    if H.load_step > 0:
        sampler = load_model(sampler, f"{H.sampler}_ema", H.load_step, H.load_dir).cuda()

    sampler = sampler.eval()

    with torch.no_grad():

        image_dir = H.log_dir

        if H.n_samples is None:

            if H.dataset == "cifar10":
                H.n_samples = 50000

            elif H.dataset == "churches":
                H.n_samples = 2500

            elif H.dataset == "ffhq":
                H.n_samples = 10000

            else:
                raise ValueError(
                    f"No default n_samples specified for dataset {H.dataset}. Please specify number of samples to \
                    calculate per step using --n_samples"
                )

        if H.stepping is not None:
            sample_stride, sample_steps = H.stepping.split("-")
            if sample_stride == "even":
                sample_steps = int(sample_steps)
            elif sample_stride == "magic":
                sample_steps = int(sample_steps)

        else:
            sample_stride, sample_steps = "all", 1000

        print(f"Sampling with temperature {H.temp}")
        all_latents = []
        for i in tqdm(range(int(H.n_samples/H.batch_size))):
            if H.sampler == "absorbing":
                if H.sample_type == "default":
                    latents = sampler.sample(temp=H.temp, sample_stride=sample_stride, sample_steps=sample_steps)
                elif H.sample_type == "v2":
                    latents = sampler.sample_v2(temp=H.temp, sample_stride=sample_stride, sample_steps=sample_steps)
            else:
                latents = sampler.sample(temp=H.temp)

            torch.save(latents.cpu(), f"logs/{image_dir}/latents_backup_{i}.pkl")
            all_latents.append(latents.cpu())

        # all_latents = [torch.load(f"logs/{image_dir}/latents_backup_{i}.pkl") for i in range(10)]
        all_latents = torch.cat(all_latents, dim=0)
        # torch.save(all_latents, f"logs/{image_dir}/all_latents_backup.pkl")
        # all_latents = torch.load(f"logs/{image_dir}/images/all_latents_backup.pkl")
        embedding_weight = sampler.embedding_weight.cuda().clone()
        # sampler = sampler.cpu()
        del sampler

        all_latents = all_latents.cuda()
        generator = generator.cuda()

        for idx, latents in tqdm(list(enumerate(torch.split(all_latents, H.vqgan_batch_size)))):
            latents_one_hot = latent_ids_to_onehot(latents, H.latent_shape, H.codebook_size).cuda()
            q = torch.matmul(latents_one_hot, embedding_weight).view(
                latents_one_hot.size(0), H.latent_shape[1], H.latent_shape[2], H.emb_dim
            ).permute(0, 3, 1, 2).contiguous()
            gen_images = generator(q)
            # vis.images(gen_images[:64].clamp(0,1), win="FID_sample_check", opts=dict(title="FID_sample_check"))
            save_images(gen_images.detach().cpu(), "sample", idx, image_dir, save_indivudally=True)
            images = BigDataset(f"logs/{image_dir}/images/")
        # generator = generator.cpu()
        del generator

        image_dir = H.log_dir
        images = BigDataset(f"logs/{image_dir}/images/")

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
