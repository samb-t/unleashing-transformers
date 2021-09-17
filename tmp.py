import torch_fidelity
import torch
import torchvision
import os

class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        return self.dataset[index][0].mul(255).clamp_(0, 255).to(torch.uint8)
    
    def __len__(self):
        return 50000
        # return len(self.dataset)

input2 = torchvision.datasets.LSUN('/home2/kmhf27/workspace/data/LSUN', classes=['church_outdoor_train'], transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor()
]))
# TODO: Maybe only compute stats for samples_needed images from the dataset?
# Yes. SOTA on churches only uses 50k https://github.com/saic-mdal/CIPS/blob/main/calculate_fid.py
# This is a good reference as it also uses torch fidelity
input2 = NoClassDataset(input2)
# input2_cache_name = 'lsun_churches'
input2_cache_name = 'lsun_churches_50k'


# directory = '/home2/kmhf27/'+'/'.join(os.getcwd().split('/')[-6:])+'/images/'

# directory = '/home2/kmhf27/workspace/2021/09/VQGAN-EBM/logs/lotta_images'
directory = '/home2/kmhf27/workspace/2021/09/VQGAN-EBM/logs/absorbing_churches_50ksamples'

metrics_dict = torch_fidelity.calculate_metrics(
    input1=directory,
    input2=input2,
    cuda=True,
    fid=True,
    verbose=True,
    input2_cache_name=input2_cache_name
)
print(metrics_dict)