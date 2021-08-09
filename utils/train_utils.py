import math

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


def optim_warmup(H, step, optim):
    if step < H.warmup_iters:
        lr = H.lr * float(step) / H.warmup_iters
        for param_group in optim.param_groups:
            param_group['lr'] = lr


def optim_warmup_cosine_decay(H, step, optim, num_training_steps=1000000, min_lr=1e-7):
    if step < H.warmup_iters:
        lr =  H.lr * float(step) / float(max(1, H.warmup_iters))
    else:
        progress = float(step - H.warmup_iters) / float(max(1, num_training_steps - H.warmup_iters))
        lr =  H.lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    for param_group in optim.param_groups:
        param_group['lr'] = max(lr, min_lr)
    

