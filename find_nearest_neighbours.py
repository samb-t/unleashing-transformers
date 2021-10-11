from utils.data_utils import get_data_loader
import lpips
import torch
from models import Generator
from hparams import get_sampler_hparams
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

    data_loader, _ = get_data_loader(H.dataset, H.img_size, H.batch_size, shuffle=False)
    
    generator.load_state_dict(quanitzer_and_generator_state_dict)
    generator = generator.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()
    sampler = load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir).cuda()

    samples = get_samples(H, generator, sampler, temp=H.temp, stride='all')
    display_images(vis, samples, H, win_name=f'Samples_{H.temp}')

    sampler = None

    distance_fn = lpips.LPIPS(net='alex').cuda()
    nearest_distances = torch.ones(H.batch_size, dtype=torch.float32).cpu() * float('inf')
    nearest_images = torch.zeros_like(samples).cpu()


    log(f'Num batches: {len(data_loader)}')
    for batch_num, image_batch in enumerate(iter(data_loader)):
        image_batch = image_batch[0].cuda()
        for idx, sample in enumerate(samples):
            for image in image_batch:
                distance = distance_fn(sample, image).item()
                
                if distance < nearest_distances[idx]:
                    nearest_distances[idx] = distance
                    nearest_images[idx] = image
        
        log(f'Batch: {batch_num}  Avg. Distance: {nearest_distances.mean().item():.4f}')

        if batch_num > 0 and batch_num % 5 == 0:
            display_images(vis, nearest_images, H, win_name='Nearest Images')

if __name__=='__main__':
    H = get_sampler_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')   
    start_training_log(H)
    main(H, vis)