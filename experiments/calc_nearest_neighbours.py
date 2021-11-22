from utils.data_utils import get_data_loaders
import lpips
import torch
from models import Generator
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, get_samples
from utils.log_utils import log, set_up_visdom, config_log, start_training_log, load_model, save_images
from train_sampler import get_sampler
from tqdm import tqdm
import torchvision


def main(H, vis):
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ["quantize", "generator"],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop(
        "embedding.weight")
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)

    data_loader, _ = get_data_loaders(
        H.dataset, H.img_size, H.batch_size, shuffle=False)

    generator.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator = generator.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()
    sampler = load_model(
        sampler, f"{H.sampler}_ema", H.load_step, H.load_dir).cuda()

    samples = get_samples(H, generator, sampler)
    sampler = None

    distance_fn = lpips.LPIPS(net="alex").cuda()
    nearest_images = torch.zeros_like(samples).cpu()

    k_nearest = 10
    nearests = [[(None, float("inf")) for _ in range(k_nearest)]
                for _ in range(H.batch_size)]

    log(f"Num batches: {len(data_loader)}")
    for batch_num, image_batch in tqdm(enumerate(iter(data_loader)), total=len(data_loader)):
        image_batch = image_batch[0].cuda()
        for idx, sample in enumerate(samples):
            for image in image_batch:
                distance = distance_fn(sample, image).item()

                nearests[idx].append((image, distance))
                nearests[idx].sort(key=lambda x: x[1])
                nearests[idx] = nearests[idx][:k_nearest]

        if batch_num > 0 and batch_num % 100 == 0:

            all_grids = []

            for idx in range(H.batch_size):
                sample = samples[idx].clamp(0, 1)
                nearest_images = [x[0] for x in nearests[idx]]

                all_images = torch.stack([sample] + nearest_images, dim=0)
                grid = torchvision.utils.make_grid(
                    all_images, nrow=all_images.size(0), padding=0)
                all_grids.append(grid)

            complete = torchvision.utils.make_grid(
                all_grids, nrow=1, padding=2)
            vis.image(complete, win="Nearest Neighbours")

    save_images(complete, "nearest_neighbours", H.load_step, H.log_dir)


if __name__ == "__main__":
    H = get_sampler_hparams()
    vis = set_up_visdom(H)
    config_log(H.log_dir)
    log("---------------------------------")
    log(f"Generating nearest neighbours for {H.sampler} model loaded from {H.load_dir}")
    start_training_log(H)
    main(H, vis)
