import torch
import random 


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class MyOneHotCategorical:
    def __init__(self, mean):
        self.mean = mean
        self.dist = torch.distributions.OneHotCategorical(probs=self.mean)

    def sample(self, x):
        return self.dist.sample(x)

    def log_prob(self, x):
        logits = self.dist.logits
        lp = torch.log_softmax(logits, -1)
        return (x * lp[None]).sum(-1)

# dataset that returns masked / swapped versions of original latents
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, latent_ids, mask_id, num_ids):
        self.latent_ids = latent_ids
        self.mask_id = mask_id
        self.num_ids = num_ids
    
    def __len__(self):
        return self.latent_ids.size(0)
    
    def __getitem__(self, item):
        latent = self.latent_ids[item].clone()
        target = []

        # num_to_mask = random.randint(1, latent.size(0))
        # block_size = 10
        # block_start_index = random.randint(0, latent.size(0)-block_size) 
        # mask = np.arange(block_start_index, block_start_index+block_size)
        # mask = np.random.choice(np.arange(latent.size(0)), size=block_size, replace=False)

        # NOTE: When calculating loss make sure to mean over dim 1 then over dim 0.

        for idx, l in enumerate(latent):
            # if idx in mask: # mask token
            #     target.append(l.clone())
            #     latent[idx] = self.mask_id 
            # elif random.random() < 0.02: # randomly change to random token
            #     target.append(l.clone())
            #     latent[idx] = random.randrange(self.num_ids)
            # else:
            #     target.append(-1)

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                target.append(l.clone())

                if prob < 0.8: # 80% randomly change token to mask token
                    latent[idx] = self.mask_id
                elif prob < 0.9: # 10% randomly change to random token
                    latent[idx] = random.randrange(self.num_ids)
                # 10% randomly don't change but use to train network
                
            else:
                # mark to not train with
                target.append(-1)
        
        target = torch.tensor(target).reshape(latent.shape).long()
        
        return latent, target