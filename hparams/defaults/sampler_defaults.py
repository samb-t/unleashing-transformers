from .base import HparamsBase


class HparamsAbsorbing(HparamsBase):
    def __init__(self, dataset):

        self.loss_type = "reweighted_elbo"
        self.sample_type = "diffusion"
        self.mask_schedule = "random"
        self.total_steps = 256
        self.sample_steps = 256
        self.attn_pdrop = 0.
        self.embd_pdrop = 0.
        self.resid_pdrop = 0.
        self.temp = 1.0

        super().__init__(dataset)
        if self.dataset == "churches" or self.dataset == "bedrooms":
            self.batch_size = 20
            self.bert_n_emb = 512
            self.bert_n_head = 8
            self.bert_n_layers = 24
            self.block_size = 256
            self.lr = 2e-4
            self.warmup_iters = 10000

        elif self.dataset == "ffhq":
            self.batch_size = 20
            self.bert_n_emb = 512
            self.bert_n_head = 8
            self.bert_n_layers = 24
            self.block_size = 256
            self.lr = 1e-4
            self.warmup_iters = 30000

        else:
            raise KeyError(f"Defaults not defined for multinomial diffusion model on dataset: {self.dataset}")


# TODO: properly configure autoregressive args
class HparamsAutoregressive(HparamsBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.attn_pdrop = 0.
        self.embd_pdrop = 0.
        self.resid_pdrop = 0.
        self.temp = 1.0

        if self.dataset == "churches" or "bedrooms":
            self.batch_size = 20
            self.bert_n_emb = 512
            self.bert_n_head = 8
            self.bert_n_layers = 24
            self.block_size = 256
            self.lr = 2e-4
            self.warmup_iters = 10000

        elif self.dataset == "ffhq":
            self.batch_size = 20
            self.bert_n_emb = 512
            self.bert_n_head = 8
            self.bert_n_layers = 24
            self.block_size = 256
            self.lr = 1e-4
            self.warmup_iters = 30000
        else:
            raise KeyError(f"Defaults not defined for BERT model on dataset: {self.dataset}")


# arguments for all sampler models
def add_sampler_args(parser):
    parser.add_argument("--ae_load_dir", type=str, required=True)
    parser.add_argument("--ae_load_step", type=int, required=True)
    parser.add_argument("--attn_pdrop", type=float)
    parser.add_argument("--bert_n_emb", type=int)
    parser.add_argument("--bert_n_head", type=int)
    parser.add_argument("--bert_n_layers", type=int)
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--embd_pdrop", type=float)
    parser.add_argument("--greedy_epochs", type=int)
    parser.add_argument("--greedy", const=True, action="store_const", default=False)
    parser.add_argument("--loss_type", type=str, choices=["reweighted_elbo", "elbo", "mlm"])
    parser.add_argument("--mask_schedule", type=str)
    parser.add_argument("--resid_pdrop", type=float)
    parser.add_argument("--sample_block_size", type=int)
    parser.add_argument("--sample_type", type=str, choices=["diffusion", "mlm"])
    parser.add_argument("--sampler", type=str, required=True, choices=["absorbing", "autoregressive"])
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--sample_steps", type=int)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--warmup_iters", type=int)
