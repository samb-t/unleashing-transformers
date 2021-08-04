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

@torch.no_grad()
def collect_recons(H, model, data_iterator):
    recons = []
    for x, *_ in data_iterator:
        x = x.cuda()
        x_hat, *_ = model.ae(x)
        recons.append(x_hat.detach().cpu())
    return torch.cat(recons, dim=0)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    
    def __getitem__(self, index):
        return self.tensor[index]
    
    def __len__(self):
        return self.tensor.size(0)

def optim_warmup(H, step, optim):
    lr = H.lr * float(step) / H.warmup_iters
    for param_group in optim.param_groups:
        param_group['lr'] = lr
