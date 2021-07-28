def optim_warmup(H, step, optim):
    if step <= H.warmup_iters:
        lr = H.lr * float(step) / H.warmup_iters
        for param_group in optim.param_groups:
            param_group['lr'] = lr