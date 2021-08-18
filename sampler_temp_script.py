from models import Generator

from hparams import get_sampler_hparams
from utils.data_utils import cycle
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts,\
                                get_samples 
from utils.log_utils import log, setup_visdom, config_log, start_training_log, \
                             load_model, save_images, display_images
from train_sampler import get_sampler



def main(H, vis):
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )
    embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight')
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)

    #NOTE: can move generator to cpu to save memory if needbe - add flag?
    generator.load_state_dict(quanitzer_and_generator_state_dict)
    generator = generator.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()

    if H.load_step > 0:
        sampler =  load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir).cuda()

    for temp_int in range(945, 955):
        temp = temp_int / 1000
        print(f'Generating samples with temp {temp}')
        samples = get_samples(H, generator, sampler, temp=temp)
        display_images(vis, samples, H, win_name=f'Samples_{temp}')


if __name__=='__main__':
    H = get_sampler_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')   
    start_training_log(H)
    main(H, vis)