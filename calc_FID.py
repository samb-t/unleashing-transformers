import torch_fidelity
import torch
from hparams import get_sampler_hparams
from models import VQGAN, Generator
from tqdm import tqdm
from train_sampler import get_sampler
import torchvision
import imageio
import os
from utils.log_utils import log, display_images, setup_visdom, load_model, save_images, config_log, start_training_log
from utils.data_utils import get_data_loader
from utils.sampler_utils import get_samples, retrieve_autoencoder_components_state_dicts, latent_ids_to_onehot

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    def __getitem__(self, index):
        return self.tensor[index]
    def __len__(self):
        return self.tensor.size(0)


class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_paths = os.listdir(folder)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = imageio.imread(self.folder+path)
        img = torch.from_numpy(img).permute(2,0,1) # -> channels first
        # How does torchvision save quantize?
        return img

    def __len__(self):
        return len(self.image_paths)


class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        self.length = length if length is not None else len(dataset)
    
    def __getitem__(self, index):
        return self.dataset[index][0].mul(255).clamp_(0, 255).to(torch.uint8)
    
    def __len__(self):
        return self.length

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
    sampler = get_sampler(H, embedding_weight).cuda()

    if H.load_step > 0:
        sampler =  load_model(sampler, f'{H.sampler}_ema', H.load_step, H.load_dir).cuda()

    sampler = sampler.eval()

    with torch.no_grad():

        image_dir = H.FID_images_dir if H.FID_images_dir is not None else f"{H.load_dir}_samples"
        
        if H.samples_needed is not None:
            samples_needed = H.samples_needed
        elif H.dataset == 'cifar10':
            samples_needed = 50000

        elif H.dataset == 'churches':
            samples_needed = 50000

        elif H.dataset == 'ffhq':
            samples_needed = 10000

        if not H.n_samples:
            raise ValueError('Please specify number of samples to calculate per step using --n_samples')

        sample_stride, sample_steps = H.stepping.split('-')
        
        if sample_stride == "dynamic":

            if sample_steps == "256useallsteps":
                # sample_steps = list(reversed([1000, 999, 998, 997, 996, 995, 994, 993, 992, 990, 989, 987, 986, 985, 981, 979, 978, 977, 976, 975, 974, 973, 972, 971, 970, 969, 968, 966, 965, 964, 963, 962, 961, 960, 958, 957, 956, 950, 949, 946, 945, 944, 943, 942, 939, 938, 937, 934, 933, 927, 926, 925, 924, 923, 921, 920, 919, 918, 911, 910, 909, 908, 906, 905, 904, 903, 898, 896, 895, 892, 890, 889, 886, 885, 884, 883, 882, 881, 880, 879, 878, 876, 875, 874, 873, 872, 870, 868, 867, 866, 865, 850, 849, 848, 847, 846, 845, 842, 841, 840, 839, 838, 837, 827, 826, 825, 818, 817, 815, 814, 812, 811, 809, 808, 807, 806, 804, 803, 802, 798, 795, 791, 790, 783, 779, 778, 777, 776, 775, 773, 772, 771, 770, 768, 767, 766, 765, 763, 762, 761, 758, 757, 756, 755, 754, 753, 752, 751, 750, 749, 748, 743, 720, 719, 718, 717, 716, 714, 712, 711, 710, 709, 708, 707, 706, 705, 698, 692, 691, 684, 680, 679, 678, 677, 676, 672, 657, 655, 654, 651, 649, 648, 646, 645, 637, 636, 635, 634, 633, 632, 631, 630, 628, 627, 626, 625, 624, 623, 614, 611, 610, 609, 608, 603, 601, 588, 585, 583, 577, 576, 573, 570, 546, 538, 537, 520, 507, 502, 501, 499, 495, 494, 489, 488, 487, 485, 484, 477, 465, 436, 434, 388, 387, 379, 335, 330, 309, 293, 281, 279, 277, 256, 233, 231, 216, 169, 142, 109, 104, 76, 61, 36, 31, 15, 4]))
                sample_steps = list(reversed([1000, 999, 998, 997, 996, 995, 994, 992, 991, 989, 988, 987, 986, 985, 984, 982, 980, 978, 977, 976, 975, 974, 972, 971, 969, 968, 966, 964, 963, 962, 961, 960, 959, 958, 957, 955, 954, 953, 952, 950, 947, 941, 940, 939, 938, 937, 936, 935, 933, 931, 930, 928, 927, 926, 924, 918, 917, 916, 915, 913, 911, 909, 907, 905, 904, 900, 899, 896, 890, 889, 888, 887, 885, 883, 878, 877, 876, 874, 871, 870, 869, 868, 867, 866, 865, 860, 859, 857, 856, 855, 854, 849, 848, 845, 841, 834, 827, 826, 824, 821, 820, 817, 810, 808, 803, 799, 791, 784, 778, 777, 776, 774, 773, 772, 767, 766, 765, 764, 762, 761, 759, 758, 757, 752, 751, 749, 748, 743, 742, 737, 736, 734, 729, 724, 722, 719, 718, 715, 707, 703, 695, 692, 691, 684, 670, 668, 664, 662, 661, 660, 653, 650, 645, 642, 640, 636, 627, 624, 620, 615, 613, 611, 605, 599, 596, 594, 592, 589, 585, 579, 567, 562, 557, 556, 552, 551, 548, 547, 541, 539, 533, 529, 524, 521, 518, 500, 486, 483, 480, 473, 466, 462, 450, 449, 439, 429, 425, 422, 412, 398, 397, 388, 387, 386, 378, 369, 368, 366, 356, 341, 336, 333, 323, 322, 296, 290, 288, 283, 282, 280, 273, 268, 257, 255, 249, 225, 218, 208, 204, 201, 198, 196, 188, 174, 173, 168, 167, 166, 164, 162, 161, 158, 151, 145, 132, 127, 122, 86, 78, 43, 38, 36, 29, 27, 9]))
            elif sample_steps == "256" or H.dynamic_stepping == "230":
                # 256 steps no infs
                sample_steps = list(reversed([1000, 999, 998, 997, 995, 994, 992, 991, 989, 988, 987, 986, 984, 982, 980, 978, 977, 976, 975, 974, 972, 971, 969, 968, 966, 964, 963, 962, 961, 960, 959, 958, 957, 955, 954, 953, 952, 950, 947, 941, 940, 938, 936, 935, 933, 931, 930, 928, 927, 926, 924, 918, 917, 915, 913, 911, 909, 907, 905, 904, 900, 899, 896, 888, 887, 885, 883, 878, 876, 874, 871, 870, 869, 868, 865, 860, 859, 857, 856, 854, 849, 848, 845, 841, 834, 827, 826, 824, 821, 820, 817, 810, 808, 803, 799, 791, 784, 777, 776, 774, 773, 772, 767, 766, 765, 762, 761, 758, 757, 752, 751, 749, 748, 743, 742, 737, 736, 734, 729, 724, 722, 719, 718, 715, 707, 703, 695, 692, 691, 684, 668, 664, 662, 661, 660, 653, 650, 645, 642, 640, 636, 627, 624, 620, 615, 613, 611, 605, 599, 594, 589, 585, 579, 567, 562, 557, 552, 551, 548, 547, 539, 533, 529, 524, 521, 518, 500, 486, 483, 480, 473, 466, 462, 449, 439, 429, 425, 422, 412, 398, 388, 387, 386, 378, 369, 368, 366, 356, 341, 336, 333, 323, 322, 296, 290, 288, 282, 280, 273, 268, 257, 255, 249, 225, 218, 208, 204, 201, 198, 196, 188, 174, 173, 167, 164, 161, 158, 151, 145, 132, 127, 122, 86, 78, 43, 38, 36, 29, 27, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            elif sample_steps == "100":
                sample_steps = list(reversed([1000, 998, 994, 987, 984, 980, 975, 971, 966, 960, 955, 950, 947, 940, 935, 930, 926, 915, 911, 904, 896, 887, 883, 876, 865, 854, 845, 841, 834, 824, 817, 810, 803, 791, 784, 772, 761, 757, 748, 742, 729, 722, 718, 715, 707, 695, 691, 684, 660, 653, 640, 636, 624, 620, 611, 605, 594, 585, 579, 567, 548, 539, 533, 518, 500, 480, 462, 449, 439, 422, 412, 398, 386, 378, 366, 356, 341, 333, 322, 290, 280, 268, 255, 249, 218, 208, 196, 188, 173, 158, 145, 127, 122, 86, 78, 38, 36, 27, 9]))
            elif sample_steps == "50":
                sample_steps = list(reversed([1000, 994, 980, 966, 955, 947, 935, 926, 913, 899, 883, 865, 854, 841, 817, 803, 784, 772, 757, 742, 718, 707, 684, 653, 636, 605, 585, 567, 548, 533, 518, 500, 480, 462, 439, 412, 386, 356, 322, 280, 268, 249, 196, 158, 122, 78, 38, 27, 9]))
            else:
                sample_steps = None
        
        elif sample_stride == "even":
            sample_steps = int(sample_steps)
        
        elif sample_stride == 'magic':
            sample_steps = int(sample_steps)
            
        else:
            sample_stride, sample_steps = 'all', 1000

        print(f'Sampling with temperature {H.temp}')
        # all_latents = []
        # for i in tqdm(range(int(samples_needed/H.batch_size) + 1)):
        #     if H.sample_type == 'default':
        #         latents = sampler.sample(temp=H.temp, sample_stride=sample_stride, sample_steps=sample_steps)
        #     elif H.sample_type == 'v2':
        #         latents = sampler.sample_v2(temp=H.temp, sample_stride=sample_stride, sample_steps=sample_steps)
        #     torch.save(latents.cpu(), f"logs/{image_dir}/latents_backup_{i}.pkl")
        #     all_latents.append(latents.cpu())

        all_latents = [torch.load(f"logs/{image_dir}/latents_backup_{i}.pkl") for i in range(137)] 
        all_latents = torch.cat(all_latents, dim=0)
        torch.save(all_latents, f"logs/{image_dir}/all_latents_backup.pkl")
        # all_latents = torch.load(f"logs/{image_dir}/all_latents_backup.pkl")
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
            # vis.images(gen_images[:64].clamp(0,1), win='FID_sample_check', opts=dict(title='FID_sample_check'))
            save_images(gen_images.detach().cpu(), f'sample', idx, image_dir, save_indivudally=True)
            # images = BigDataset(f"logs/{image_dir}/images/")
        # generator = generator.cpu()
        del generator

        images = BigDataset(f"logs/{image_dir}/images/")
        

        if H.dataset == 'cifar10':
            input2 = 'cifar10-train'
            input2_cache_name = 'cifar10-train'
        elif H.dataset == 'churches':
            input2 = torchvision.datasets.LSUN('../../../data/LSUN', classes=['church_outdoor_train'], transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(256),
                torchvision.transforms.ToTensor()
            ]))
            input2 = NoClassDataset(input2)
            input2_cache_name = 'lsun_churches'
        elif H.dataset == 'bedrooms':
            input2 = torchvision.datasets.LSUN('/projects/cgw/lsun', classes=['bedroom_train'], transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(256),
                torchvision.transforms.ToTensor()
            ]))
            input2 = NoClassDataset(input2, length=50000)
            input2_cache_name = 'lsun_bedroom_train_50k'
        elif H.dataset == 'ffhq':
            input2 = torchvision.datasets.ImageFolder('../../../data/FFHQ-256',  transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor()
            ]))
            input2 = NoClassDataset(input2)
            input2_cache_name = 'ffhq'

        metrics_dict = torch_fidelity.calculate_metrics(
            input1=images,
            input2=input2,
            cuda=True,
            fid=True,
            verbose=True,
            input2_cache_name=input2_cache_name
        )
        log(metrics_dict)


if __name__=='__main__':
    H = get_sampler_hparams()
    H.vqgan_batch_size = 32
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    if H.load_step > 0:
        log(f'Calculating FID for {H.model} loaded from step {H.load_step}')  
        start_training_log(H)
        main(H, vis) 
    else:
        raise ValueError('No value provided for load_step, cannot calculate FID for new model')
