import torch


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
