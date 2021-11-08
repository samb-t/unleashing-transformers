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
from tqdm import tqdm
import torchvision
import numpy as np
import os
import imageio

class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder, length=None):
        self.folder = folder
        self.image_paths = os.listdir(folder)
        self.length = length if length is not None else len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = imageio.imread(self.folder+path)
        img = torch.from_numpy(img).permute(2,0,1) # -> channels first
        # How does torchvision save quantize?
        return img

    def __len__(self):
        return self.length

def main(H, vis):
    # quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
    #     H,
    #     ['quantize', 'generator'],
    #     remove_component_from_key=True
    # )
    # embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight')
    # embedding_weight = embedding_weight.cuda()
    # generator = Generator(H)

    # data_loader, _ = get_data_loader(H.dataset, H.img_size, H.batch_size, shuffle=False)

    train_dataset = torchvision.datasets.LSUN('../../../data/LSUN', classes=['church_outdoor_train'], transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(256),
                    torchvision.transforms.ToTensor()
                ]))
    data_loader = torch.utils.data.DataLoader(train_dataset, num_workers=2, sampler=None, shuffle=False, batch_size=H.batch_size, drop_last=False)
    
    # generator.load_state_dict(quanitzer_and_generator_state_dict)
    # generator = generator.cuda()
    # sampler = get_sampler(H, embedding_weight).cuda()
    # sampler = load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir).cuda()

    # 100 steps
    # sample_steps = list(reversed([1000, 999, 998, 997, 996, 995, 992, 989, 985, 981, 975, 971, 968, 965, 956, 939, 937, 934, 933, 926, 921, 920, 918, 908, 906, 903, 890, 889, 880, 873, 870, 867, 866, 865, 838, 837, 825, 806, 803, 802, 798, 795, 790, 775, 765, 763, 761, 748, 743, 708, 707, 705, 698, 691, 684, 676, 672, 654, 648, 630, 625, 624, 623, 614, 611, 608, 601, 585, 576, 573, 570, 546, 537, 520, 507, 484, 465, 436, 434, 387, 379, 335, 330, 309, 293, 279, 277, 256, 231, 216, 169, 142, 104, 76, 61, 36, 31, 15, 4]))
    
    # 10 steps
    # sample_steps = list(reversed([1000, 956, 865, 748, 672, 537, 379, 216, 36]))

    # 256 steps
    # sample_steps = list(reversed([1000, 999, 998, 997, 996, 995, 994, 992, 991, 989, 988, 987, 986, 985, 984, 982, 980, 978, 977, 976, 975, 974, 972, 971, 969, 968, 966, 964, 963, 962, 961, 960, 959, 958, 957, 955, 954, 953, 952, 950, 947, 941, 940, 939, 938, 937, 936, 935, 933, 931, 930, 928, 927, 926, 924, 918, 917, 916, 915, 913, 911, 909, 907, 905, 904, 900, 899, 896, 890, 889, 888, 887, 885, 883, 878, 877, 876, 874, 871, 870, 869, 868, 867, 866, 865, 860, 859, 857, 856, 855, 854, 849, 848, 845, 841, 834, 827, 826, 824, 821, 820, 817, 810, 808, 803, 799, 791, 784, 778, 777, 776, 774, 773, 772, 767, 766, 765, 764, 762, 761, 759, 758, 757, 752, 751, 749, 748, 743, 742, 737, 736, 734, 729, 724, 722, 719, 718, 715, 707, 703, 695, 692, 691, 684, 670, 668, 664, 662, 661, 660, 653, 650, 645, 642, 640, 636, 627, 624, 620, 615, 613, 611, 605, 599, 596, 594, 592, 589, 585, 579, 567, 562, 557, 556, 552, 551, 548, 547, 541, 539, 533, 529, 524, 521, 518, 500, 486, 483, 480, 473, 466, 462, 450, 449, 439, 429, 425, 422, 412, 398, 397, 388, 387, 386, 378, 369, 368, 366, 356, 341, 336, 333, 323, 322, 296, 290, 288, 283, 282, 280, 273, 268, 257, 255, 249, 225, 218, 208, 204, 201, 198, 196, 188, 174, 173, 168, 167, 166, 164, 162, 161, 158, 151, 145, 132, 127, 122, 86, 78, 43, 38, 36, 29, 27, 9]))

    # 256 steps no infs
    # sample_steps = list(reversed([1000, 999, 998, 997, 995, 994, 992, 991, 989, 988, 987, 986, 984, 982, 980, 978, 977, 976, 975, 974, 972, 971, 969, 968, 966, 964, 963, 962, 961, 960, 959, 958, 957, 955, 954, 953, 952, 950, 947, 941, 940, 938, 936, 935, 933, 931, 930, 928, 927, 926, 924, 918, 917, 915, 913, 911, 909, 907, 905, 904, 900, 899, 896, 888, 887, 885, 883, 878, 876, 874, 871, 870, 869, 868, 865, 860, 859, 857, 856, 854, 849, 848, 845, 841, 834, 827, 826, 824, 821, 820, 817, 810, 808, 803, 799, 791, 784, 777, 776, 774, 773, 772, 767, 766, 765, 762, 761, 758, 757, 752, 751, 749, 748, 743, 742, 737, 736, 734, 729, 724, 722, 719, 718, 715, 707, 703, 695, 692, 691, 684, 668, 664, 662, 661, 660, 653, 650, 645, 642, 640, 636, 627, 624, 620, 615, 613, 611, 605, 599, 594, 589, 585, 579, 567, 562, 557, 552, 551, 548, 547, 539, 533, 529, 524, 521, 518, 500, 486, 483, 480, 473, 466, 462, 449, 439, 429, 425, 422, 412, 398, 388, 387, 386, 378, 369, 368, 366, 356, 341, 336, 333, 323, 322, 296, 290, 288, 282, 280, 273, 268, 257, 255, 249, 225, 218, 208, 204, 201, 198, 196, 188, 174, 173, 167, 164, 161, 158, 151, 145, 132, 127, 122, 86, 78, 43, 38, 36, 29, 27, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    # 100 steps
    # sample_steps = list(reversed([1000, 998, 994, 987, 984, 980, 975, 971, 966, 960, 955, 950, 947, 940, 935, 930, 926, 915, 911, 904, 896, 887, 883, 876, 865, 854, 845, 841, 834, 824, 817, 810, 803, 791, 784, 772, 761, 757, 748, 742, 729, 722, 718, 715, 707, 695, 691, 684, 660, 653, 640, 636, 624, 620, 611, 605, 594, 585, 579, 567, 548, 539, 533, 518, 500, 480, 462, 449, 439, 422, 412, 398, 386, 378, 366, 356, 341, 333, 322, 290, 280, 268, 255, 249, 218, 208, 196, 188, 173, 158, 145, 127, 122, 86, 78, 38, 36, 27, 9]))

    # # samples = get_samples(H, generator, sampler, temp=1.0, stride='dynamic', sample_steps=sample_steps)
    # samples = get_samples(H, generator, sampler, temp=1.0, stride='all')
    # display_images(vis, samples, H, win_name='skipped-sampling')
    # exit()

    # for i in range(100):
    #     samples = get_samples(H, generator, sampler, temp=H.temp)
    #     image = torchvision.utils.make_grid(samples.clamp(0,1), nrow=int(np.sqrt(samples.size(0))), padding=0)
    #     torchvision.utils.save_image(samples.clamp(0,1), f'/home/sam/Documents/VQGAN-EBM_Figures/grouped_samples/samples_{i}.png', nrow=int(np.sqrt(samples.size(0))), padding=0)
    # exit()

    all_samples = BigDataset(f"logs/nice_church_samples/", length=18)

    # sampler = None

    # nice_samples = [13, ]
    samples = torch.stack([x for x in all_samples], dim=0).cuda().float() / 255

    distance_fn = lpips.LPIPS(net='alex').cuda()
    nearest_distances = torch.ones(H.batch_size, dtype=torch.float32).cpu() * float('inf')
    nearest_images = torch.zeros_like(samples).cpu()

    k_nearest = 10
    nearests = [[(None, float('inf')) for _ in range(k_nearest)] for _ in range(H.batch_size)]

    log(f'Num batches: {len(data_loader)}')
    for batch_num, image_batch in tqdm(enumerate(iter(data_loader)), total=len(data_loader)):
        image_batch = image_batch[0].cuda()
        for idx, sample in enumerate(samples):
            for image in image_batch:
                distance = distance_fn(sample, image).item()

                # really not efficient but eh
                nearests[idx].append((image, distance))
                nearests[idx].sort(key=lambda x: x[1])
                nearests[idx] = nearests[idx][:k_nearest]
                
                # if distance < nearest_distances[idx]:
                #     nearest_distances[idx] = distance
                #     nearest_images[idx] = image
        
        # log(f'Batch: {batch_num}  Avg. Distance: {nearest_distances.mean().item():.4f}')

        if (batch_num > 0 and batch_num % 100 == 0) or (batch_num == len(data_loader)-1):

            all_grids = []
            
            for idx in range(H.batch_size):
                sample = samples[idx].clamp(0,1)
                nearest_images = [x[0] for x in nearests[idx]]

                # print(sample.shape, [x.shape for x in nearest_images])

                all_images = torch.stack([sample] + nearest_images, dim=0)
                grid = torchvision.utils.make_grid(all_images, nrow=all_images.size(0), padding=0)
                all_grids.append(grid)

            complete = torchvision.utils.make_grid(all_grids, nrow=1, padding=2)
            vis.image(complete, win='nearests')

            # display_images(vis, nearest_images, H, win_name='Nearest Images')

if __name__=='__main__':
    H = get_sampler_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')   
    start_training_log(H)
    main(H, vis)