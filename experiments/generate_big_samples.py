import torch

from hparams import get_big_samples_hparams
from models import Generator
from train_sampler import get_sampler
from utils.log_utils import (config_log, load_model, log, set_up_visdom, start_training_log)
from utils.sampler_utils import (latent_ids_to_onehot, retrieve_autoencoder_components_state_dicts)


def main(H, vis):

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight')
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)
    generator.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator = generator.cuda()

    model = get_sampler(H, embedding_weight)
    model = load_model(model, H.sampler + '_ema', H.load_step, H.load_dir).cuda()
    model = model.eval()

    shape = (1, H.shape[0], H.shape[1])
    log(f"Generating latents of shape: {shape}")
    step = 1
    time_steps = shape[1] * shape[2]

    with torch.no_grad():
        latents = model.sample_shape(shape[1:], H.batch_size, time_steps=time_steps, step=step)
        latents_one_hot = latent_ids_to_onehot(latents, shape, H.codebook_size).cuda()

        q = torch.matmul(latents_one_hot, embedding_weight).view(
            latents_one_hot.size(0), shape[1], shape[2], H.emb_dim
        ).permute(0, 3, 1, 2).contiguous()
        all_images = []
        del model
        for image_latents in torch.split(q, 8):
            all_images.append(generator(image_latents))
        gen_images = torch.cat(all_images, dim=0)
        vis.images(gen_images.clamp(0, 1), win='large_samples', opts=dict(title='large_samples'))


if __name__ == '__main__':
    H = get_big_samples_hparams()
    config_log(H.log_dir)
    vis = set_up_visdom(H)
    log('---------------------------------')
    if H.load_step > 0:
        # log(f'Calculating FID for {H.sampler} loaded from step {H.load_step}')  what????
        start_training_log(H)
        main(H, vis)
    else:
        raise ValueError('No value provided for load_step, cannot calculate FID for new model')
