
from models import Generator
from hparams import get_sampler_hparams
from utils.log_utils import save_images, set_up_visdom, config_log, log, start_training_log, display_images, load_model
from utils.sampler_utils import get_sampler, get_samples, retrieve_autoencoder_components_state_dicts


def main(H, vis):

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )

    embedding_weight = quanitzer_and_generator_state_dict.pop(
        'embedding.weight')
    if H.deepspeed:
        embedding_weight = embedding_weight.half()
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)

    generator.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator = generator.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()

    sampler = load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir)
    sampler.n_samples = 25  # get samples in 5x5 grid

    for i in range(100):
        images = get_samples(H, generator, sampler, temp=H.temp, stride="magic", sample_steps=256)
        display_images(vis, images, H, win_name=f'{H.sampler}_samples')
        save_images(images, "samples", i, H.log_dir)


if __name__ == '__main__':
    H = get_sampler_hparams()
    vis = set_up_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)
    main(H, vis)
