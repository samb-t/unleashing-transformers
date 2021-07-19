import argparse
from numpy import add

from torch.utils.data import dataset

class Hparams(dict):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

    def __str__(self):
        return f'H object of dataset: {self.dataset}'


class HparamsVQGAN(Hparams):
    def __init__(self, dataset):
        super().__init__(dataset)
        if self.dataset == 'mnist':
            # vqgan architecture defaults
            self.img_size = 32
            self.n_channels = 1
            self.nf = 32
            self.ndf = 16
            self.ch_mult = [1,1,1]
            self.attn_resolutions = [8]
            self.res_blocks = 1
            self.disc_layers = 1
            self.codebook_size = 10
            self.emb_dim = 64
            self.latent_shape = [1, 8, 8]


            # vqgan training defaults
            self.vqgan_batch_size = 128
            self.perceptual_weight = 0.0
            self.disc_start_step = 2001
            self.vq_base_lr = 4.5e-6

        elif self.dataset == 'cifar10':
            # vqgan architecture defaults
            self.img_size = 32
            self.n_channels = 3
            self.nf = 64
            self.ndf = 32
            self.ch_mult = [1,1,2]
            self.attn_resolutions = [8]
            self.res_blocks = 1
            self.disc_layers = 1
            self.codebook_size = 128
            self.emb_dim = 256
            self.latent_shape = [1, 8, 8]

            # vqgan training defaults
            self.vqgan_batch_size = 128
            self.perceptual_weight = 1.0
            self.disc_start_step = 10001
            self.vq_base_lr = 4.5e-6

        elif self.dataset == 'flowers':
            # vqgan architecture defaults
            self.img_size = 32
            self.n_channels = 3
            self.nf = 64
            self.ndf = 32
            self.ch_mult = [1,1,2]
            self.attn_resolutions = [8]
            self.res_blocks = 1
            self.disc_layers = 1
            self.codebook_size = 128
            self.emb_dim = 256
            self.latent_shape = [1, 8, 8]

            # vqgan training defaults
            self.vqgan_batch_size = 128
            self.perceptual_weight = 1.0
            self.disc_start_step = 10001
            self.vq_base_lr = 4.5e-6

        elif self.dataset == 'churches':
            # vqgan architecture defaults
            self.img_size = 256
            self.n_channels = 3
            self.nf = 128
            self.ndf = 64
            self.ch_mult = [1, 1, 2, 2, 4]
            self.attn_resolutions = [16]
            self.res_blocks = 2
            self.disc_layers = 3
            self.codebook_size = 1024
            self.emb_dim = 256
            self.latent_shape = [1, 16, 16]

            # vqgan training defaults
            self.vqgan_batch_size = 3
            self.perceptual_weight = 1.0
            self.disc_start_step = 30001
            self.vq_base_lr = 4.5e-6 

        elif self.dataset == 'celeba' or self.dataset == 'ffhq':
            # vqgan architecture defaults
            self.img_size = 256
            self.n_channels = 3
            self.nf = 128
            self.ndf = 64
            self.ch_mult = [1, 1, 2, 2, 4]
            self.attn_resolutions = [16]
            self.res_blocks = 2
            self.disc_layers = 3
            self.codebook_size = 1024
            self.emb_dim = 256
            self.latent_shape = [1, 8, 8]

            # vqgan training defaults
            self.vqgan_batch_size = 3
            self.perceptual_weight = 1.0
            self.disc_start_step = 30001
            self.vq_base_lr = 4.5e-6 

        elif self.dataset == None:
            raise KeyError('Please specify a dataset using the -d flag')
        else:
            raise KeyError(f'Defaults not defined for VQGAN model on dataset: {self.dataset}')

        def get_vqgan_param_dict(self):
            return dict(
                img_size=self.img_size,
                n_channels=self.n_channels,
                nf=self.nf,
                ndf=self.ndf,
                ch_mult=self.ch_mult,
                attn_resolutions=self.attn_resolutions,
                res_blocks=self.res_blocks,
                disc_layers=self.disc_layers,
                codebook_size=self.codebook_size,
                emb_dim=self.emb_dim,
                latent_shape=self.latent_shape,
                vqgan_batch_size=self.vqgan_batch_size, 
                perceptual_weight=self.perceptual_weight, 
                disc_start_step=self.disc_start_step,
                vq_base_lr=self.vq_base_lr 
            )


    def set_vqgan_lr(self):
        self.vqgan_lr = self.vqgan_batch_size * self.vq_base_lr

    def get_vqgan_param_dict(self):
        return dict(
            dataset = self.dataset,
            batch_size = self.vqgan_batch_size,
            lr = self.vqgan_lr,
            img_size = self.img_size,
            n_channels = self.n_channels,
            nf=self.nf,
            ndf=self.ndf,
            ch_mult=self.ch_mult,
            attn_resolutions=self.attn_resolutions,
            res_blocks=self.res_blocks,
            disc_layers=self.disc_layers,
            codebook_size = self.codebook_size,
            emb_dim = self.emb_dim,
            latent_shape = self.latent_shape,
            perceptual_weight=self.perceptual_weight,
            disc_start_step=self.disc_start_step
        )


class HparamsEBM(Hparams):
    def __init__(self, dataset):
        super().__init__(dataset)
        if self.dataset == 'mnist':
            # ebm architcture defaults
            self.block_str = 'rdrdr'

            # ebm training defaults
            self.batch_size = 128
            self.buffer_size = 10000
            self.mcmc_steps = 50
            self.warmup_iters = 2000
            self.lr = 1e-4
            self.l2_coef = 0
            self.grad_clip_threshold = 1000

        elif self.dataset == 'cifar10':
            # ebm architecture defaults
            self.block_str = 'rdrrdrr'

            # ebm training defaults
            self.batch_size = 128
            self.buffer_size = 10000
            self.mcmc_steps = 50
            self.warmup_iters = 2000
            self.lr = 1e-5
            self.l2_coef = 0
            self.grad_clip_threshold = 1000

        elif self.dataset == 'flowers':
            # ebm architecture defaults
            self.block_str = 'rdrrdrr'

            # ebm training defaults
            self.batch_size = 128
            self.buffer_size = 10000
            self.mcmc_steps = 50
            self.warmup_iters = 2000
            self.lr = 1e-4
            self.l2_coef = 0
            self.grad_clip_threshold = 1000

        elif self.dataset == 'churches':
            ...

        elif self.dataset == 'celeba' or self.dataset == 'ffhq':
            # ebm archtiecture defaults
            self.block_str = 'rdrrdrrdrr'

            # ebm training defaults
            self.batch_size = 32
            self.buffer_size = 10000
            self.mcmc_steps = 50
            self.warmup_iters = 2000
            self.lr = 5e-6
            self.l2_coef = 0
            self.grad_clip_threshold = 10000

        elif self.dataset == None:
            raise KeyError('Please specify a dataset using the -d flag')
        else:
            raise KeyError(f'Defaults not defined for EBM model on dataset: {self.dataset}')


class HparamsBERT(Hparams):
    def __init__(self, dataset):
        super().__init__(dataset)
        if self.dataset == 'mnist':
            # bert architcture defaults
            self.block_size = 512
            self.bert_n_layers = 16
            self.bert_n_head = 8
            self.bert_n_emb = 512

            # bert training defaults
            self.batch_size = 32
            self.lr = 1e-4
            self.sample_block_size = 1

        elif self.dataset == 'cifar10':
            # bert architcture defaults
            self.block_size = 512
            self.bert_n_layers = 16
            self.bert_n_head = 8
            self.bert_n_emb = 512

            # bert training defaults
            self.batch_size = 32
            self.lr = 1e-4
            self.sample_block_size = 1

        elif self.dataset == 'flowers':
            ...

        elif self.dataset == 'churches':

            # bert architcture defaults
            self.block_size = 256
            self.bert_n_layers = 8
            self.bert_n_head = 8
            self.bert_n_emb = 256

            # bert training defaults
            self.batch_size = 32
            self.lr = 1e-4
            self.sample_block_size = 1

        elif self.dataset == 'celeba' or self.dataset == 'ffhq':
            ...

        elif self.dataset == None:
            raise KeyError('Please specify a dataset using the -d flag')
        else:
            raise KeyError(f'Defaults not defined for BERT model on dataset: {self.dataset}')

    
class HparamsMultinomialDiffusion(Hparams):
    def __init__(self, dataset):
        super().__init__(dataset)
        if self.dataset == 'mnist':
            ...

        elif self.dataset == 'cifar10':
            self.batch_size = 128
            self.lr = 1e-3
            self.diffusion_steps = 1000
            self.warmup_iters = 2500 # approx 5 epochs with bs = 128
            self.unet_dim = 32
            self.unet_dim_mults = [1,2,4,8]
            
        elif self.dataset == 'flowers':
            ...

        elif self.dataset == 'churches':
            ...

        elif self.dataset == 'celeba' or self.dataset == 'ffhq':
            ...

        elif self.dataset == None:
            raise KeyError('Please specify a dataset using the -d flag')
        else:
            raise KeyError(f'Defaults not defined for multinomial diffusion model on dataset: {self.dataset}')


def add_training_args(parser):
    parser.add_argument('--model', type=str, default='vqgan')
    parser.add_argument('--dataset', type=str)

    # training loop control args
    parser.add_argument('--train_steps', type=int, default=1000000)
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--load_optim', const=True, action='store_const', default=False)

    # logging args
    parser.add_argument('--log_dir', type=str, default='test')
    parser.add_argument('--visdom_port', type=int, default=8097)
    parser.add_argument('--ncc', const=True, action='store_const', default=False)
    parser.add_argument('--steps_per_log', type=int, default=1)
    parser.add_argument('--steps_per_checkpoint', type=int, default=1000)
    parser.add_argument('--steps_per_display_output', type=int, default=50)
    parser.add_argument('--steps_per_save_output', type=int, default=100)

def add_vqgan_args(parser):
    ## training
    parser.add_argument('--vqgan_batch_size', type=int)
    parser.add_argument('--perceptual_weight', type=int)
    parser.add_argument('--disc_start_step', type=int)
    parser.add_argument('--vq_base_lr', type=float)

    ## architecture
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--n_channels', type=int)
    parser.add_argument('--nf', type=int)
    parser.add_argument('--ndf', type=int)
    parser.add_argument('--ch_mult', nargs='+', type=int)
    parser.add_argument('--attn_resolutions', nargs='+', type=int)
    parser.add_argument('--res_blocks', type=int)
    parser.add_argument('--disc_layers', type=int)
    parser.add_argument('--codebook_size', type=int)
    parser.add_argument('--emb_dim', type=int)
    parser.add_argument('--latent_shape', nargs='+', type=int)


# arguments for all sampler models
def add_sampler_args(parser):
    parser.add_argument('--ae_load_step', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--n_samples', type=int, default=16)
   

def add_ebm_args(parser):
    # architecture args
    parser.add_argument('--block_str', type=str)

    # training args
    parser.add_argument('--buffer_size', type=int)
    parser.add_argument('--mcmc_steps', type=int)
    parser.add_argument('--warmup_iters', type=int)
    parser.add_argument('--l2_coef', type=float)
    parser.add_argument('--reinit_buffer_prob', type=float)
    parser.add_argument('--grad_clip_threshold', type=int)


def add_bert_args(parser):
    # architecture args
    parser.add_argument('--block_size', type=int)
    parser.add_argument('--embd_pdrop', type=float, default=0.)
    parser.add_argument('--resid_pdrop', type=float, default=0.)    
    parser.add_argument('--attn_pdrop', type=float, default=0.)
    parser.add_argument('--bert_n_layers', type=int)
    parser.add_argument('--bert_n_head', type=int)
    parser.add_argument('--bert_n_emb', type=int)

    # training args
    parser.add_argument('--sample_block_size', type=int)
    parser.add_argument('--greedy', const=True, action='store_const', default=False)
    parser.add_argument('--greedy_epochs', type=int, default=25)


def add_diffusion_args(parser):
    # architecture args
    parser.add_argument('--diffusion_net', type=str, default='unet')
    parser.add_argument('--unet_dim', type=int)
    parser.add_argument('--unet_dim_mults', nargs='+', type=int)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--groups', type=int, default=8)

    # training args
    parser.add_argument('--diffusion_loss', type=str, default='vb_stochastic')
    parser.add_argument('--diffusion_steps', type=int)
    parser.add_argument('--parametrization', type=str, default='x0')

def get_hparams():
    parser = argparse.ArgumentParser(description='Arguments for training stuff :)')

    add_training_args(parser)
    add_vqgan_args(parser)
    add_sampler_args(parser)
    add_ebm_args(parser)
    add_bert_args(parser)
    add_diffusion_args(parser)

    # parse arguments and load defaults

    args = parser.parse_args()
    model = args.model
    dataset = args.dataset

    # load defaults for given model and dataset
    H = HparamsVQGAN(dataset)

    if model == 'vqgan':
        sampler_H = None
    elif model == 'ebm':
        sampler_H = HparamsEBM(dataset)
    elif model == 'bert':
        sampler_H = HparamsBERT(dataset)
    elif model == 'diffusion':
        sampler_H = HparamsMultinomialDiffusion(dataset)
    
    # add sampler
    if sampler_H != None:
        H.update(sampler_H)

    args = args.__dict__
    for arg in args:
        if args[arg] != None:
            H[arg] = args[arg]

    H.set_vqgan_lr()

    assert H.steps_per_save_output % H.steps_per_display_output == 0
    return H


    ''' hparams to add:
    - ae_load_dir
    - load_dir
    - n_samples
    - AIS_iters
    - steps_per_iter

    '''