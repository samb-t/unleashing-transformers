from .base import HparamsBase


class HparamsAbsorbing(HparamsBase):
    def __init__(self, dataset):

        self.loss_type = 'elbo'
        self.attn_pdrop = 0.
        self.embd_pdrop = 0.
        self.resid_pdrop = 0.
        super().__init__(dataset)
        if self.dataset == 'mnist':
            self.batch_size = 128
            self.bert_n_emb = 128
            self.bert_n_head = 8
            self.bert_n_layers = 4
            self.block_size = 128
            self.diffusion_steps = 1000
            self.lr = 1e-4
            self.n_samples = 128

        elif self.dataset == 'cifar10':
            self.batch_size = 32
            self.bert_n_emb = 256
            self.bert_n_head = 16
            self.bert_n_layers = 24
            self.block_size = 256
            self.diffusion_steps = 1000
            self.lr = 1e-4
            self.n_samples = 64
            self.warmup_iters = 5000
            self.loss_type = 'normed'

            self.unet_dim = 32
            self.unet_dim_mults = [1,2,4,8]
            self.dropout = 0.0
            
        elif self.dataset == 'flowers':
            self.batch_size = 32
            self.bert_n_emb = 256
            self.bert_n_head = 16
            self.bert_n_layers = 24
            self.block_size = 256
            self.diffusion_steps = 1000
            self.lr = 1e-4
            self.n_samples = 64

        elif self.dataset == 'churches':
            self.batch_size = 32
            self.bert_n_emb = 256
            self.bert_n_head = 16
            self.bert_n_layers = 24
            self.block_size = 256
            self.diffusion_steps = 1000
            self.lr = 1e-4
            self.n_samples = 16

        elif self.dataset == 'celeba' or self.dataset == 'ffhq':
            ...    

        else:
            raise KeyError(f'Defaults not defined for multinomial diffusion model on dataset: {self.dataset}')

# TODO: properly configure autoregressive args
class HparamsAutoregressive(HparamsBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.attn_pdrop = 0.
        self.embd_pdrop = 0.
        self.resid_pdrop = 0.

        if self.dataset == 'mnist':
            self.batch_size = 32
            self.bert_n_emb = 512
            self.bert_n_head = 8
            self.bert_n_layers = 16
            self.block_size = 512
            self.lr = 1e-4
            self.sample_block_size = 1

        elif self.dataset == 'cifar10':
            self.batch_size = 64
            self.bert_n_emb = 256
            self.bert_n_head = 16
            self.bert_n_layers = 24
            self.block_size = 64
            self.lr = 5e-4
            self.sample_block_size = 1
            self.warmup_iters = 5000

        elif self.dataset == 'flowers':
            ...

        elif self.dataset == 'churches':
            self.batch_size = 32
            self.bert_n_emb = 256
            self.bert_n_head = 8
            self.bert_n_layers = 8
            self.block_size = 256
            self.lr = 1e-4
            self.sample_block_size = 1

        elif self.dataset == 'celeba' or self.dataset == 'ffhq':
            self.batch_size = 32
            self.bert_n_emb = 256
            self.bert_n_head = 8
            self.bert_n_layers = 8
            self.block_size = 256
            self.lr = 1e-4
            self.sample_block_size = 1
        else:
            raise KeyError(f'Defaults not defined for BERT model on dataset: {self.dataset}')

# TODO: properly configure Multinomial diffusion args
class HparamsMultinomialDiffusion(HparamsBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        if self.dataset == 'mnist':
            ...

        elif self.dataset == 'cifar10':
            self.batch_size = 128
            self.diffusion_steps = 1000
            self.lr = 1e-3
            self.unet_dim = 32
            self.unet_dim_mults = [1,2,4,8]
            self.warmup_iters = 2500 # approx 5 epochs with bs = 128
            
        elif self.dataset == 'flowers':
            ...

        elif self.dataset == 'churches':
            ...

        elif self.dataset == 'celeba' or self.dataset == 'ffhq':
            ...

        else:
            raise KeyError(f'Defaults not defined for multinomial diffusion model on dataset: {self.dataset}')


def add_bert_args(parser):
    parser.add_argument('--attn_pdrop', type=float)
    parser.add_argument('--bert_n_emb', type=int)
    parser.add_argument('--bert_n_head', type=int)
    parser.add_argument('--bert_n_layers', type=int)
    parser.add_argument('--block_size', type=int)
    parser.add_argument('--embd_pdrop', type=float)
    parser.add_argument('--greedy_epochs', type=int)
    parser.add_argument('--greedy', const=True, action='store_const', default=False)
    parser.add_argument('--resid_pdrop', type=float)    
    parser.add_argument('--sample_block_size', type=int)


def add_diffusion_args(parser):
    parser.add_argument('--diffusion_loss', type=str)#, default='vb_stochastic')
    parser.add_argument('--diffusion_net', type=str)#, default='unet')
    parser.add_argument('--diffusion_steps', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--groups', type=int, default=8)
    parser.add_argument('--parametrization', type=str, default='x0')
    parser.add_argument('--unet_dim_mults', nargs='+', type=int)
    parser.add_argument('--unet_dim', type=int)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--mask_schedule', type=str, default='random')


def add_ebm_args(parser):
    parser.add_argument('--block_str', type=str)
    parser.add_argument('--buffer_size', type=int)
    parser.add_argument('--grad_clip_threshold', type=int)
    parser.add_argument('--l2_coef', type=float)
    parser.add_argument('--mcmc_steps', type=int)
    parser.add_argument('--reinit_buffer_prob', type=float)


# arguments for all sampler models
def add_sampler_args(parser):
    parser.add_argument('--ae_load_dir', type=str, required=True)
    parser.add_argument('--ae_load_step', type=int, required=True)
    parser.add_argument('--sampler', type=str, required=True)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--warmup_iters', type=int)

    add_bert_args(parser)
    add_diffusion_args(parser)
    add_ebm_args(parser)