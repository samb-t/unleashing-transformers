import torch_fidelity
import torch
from hparams import get_sampler_hparams
from utils.log_utils import log, config_log, start_training_log
from utils.experiment_utils import generate_images_from_latents, generate_samples, get_generator_and_embedding_weight
from utils.data_utils import BigDataset, NoClassDataset, get_datasets


def main(H):
    real_dataset, _ = get_datasets(H.dataset, H.img_size, custom_dataset_path=H.custom_dataset_path)
    real_dataset = NoClassDataset(real_dataset)

    if not H.latents_path:
        log(f"Generating {H.n_samples} samples for {H.dataset} dataset")
        generate_samples(H)
    else:
        log(f"Loading latents from {H.latents_path}")
        latents = torch.load(H.latents_path)
        log("Generating samples from provided latents")
        generator, embedding_weight = get_generator_and_embedding_weight(H)
        generate_images_from_latents(H, latents, embedding_weight, generator)

    fake_images_path = f"logs/{H.log_dir}/images/"
    fake_dataset = BigDataset(fake_images_path)

    log("Calculating FID metrics")
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=fake_dataset,
        input2=real_dataset,
        cuda=True,
        fid=True,
        verbose=True,
        input2_cache_name=f"{H.dataset}_cache" if H.dataset != "custom" else None,
    )
    log(metrics_dict)


if __name__ == "__main__":
    H = get_sampler_hparams(get_FID_args=True)
    if H.log_dir == "test":  # i.e. if it hasn"t been set using a flag)
        H.log_dir = f"{H.load_dir}_FID_samples"
    config_log(H.log_dir)
    log("---------------------------------")
    if H.load_step > 0:
        log(f"Calculating FID for {H.model} loaded from step {H.load_step}")
        start_training_log(H)
        main(H)
    else:
        raise ValueError("No value provided for --load_step, cannot calculate FID for new model")
