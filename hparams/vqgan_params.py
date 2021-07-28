from hparams import Hparams

class HparamsVQGAN(Hparams):
    def __init__(self, dataset):
        super().__init__(dataset)

        # default that are same for all datasets
        self.vq_base_lr = 4.5e-6
        
        # quantizers
        self.quantizer = 'nearest'
        self.beta = 0.25 # for nearest quantizer
        self.gumbel_straight_through = False
        self.gumbel_kl_weight = 1e-8

        # diffaug defaults
        self.diff_aug = False
        
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
            self.disc_start_step = 30001
            

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
            self.latent_shape = [1, 16, 16]

            # vqgan training defaults
            self.vqgan_batch_size = 3
            self.perceptual_weight = 1.0
            self.disc_start_step = 30001

        else:
            raise KeyError(f'Defaults not defined for VQGAN model on dataset: {self.dataset}')


def add_vqgan_args(parser):
    parser.add_argument('--perceptual_weight', type=int)
    parser.add_argument('--disc_start_step', type=int)
    parser.add_argument('--vq_base_lr', type=float)
    parser.add_argument('--steps_per_fid_calc', type=int)

    # architecture
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
    parser.add_argument('--quantizer', type=str)

    # nearest quantizer 
    parser.add_argument('--beta', type=float)
    
    # gumbel quantizer
    parser.add_argument('--gumbel_straight_through', const=True, action='store_const', default=False)
    parser.add_argument('--gumbel_kl_weight', type=float)

    # diffaug
    parser.add_argument('--diff_aug', const=True, action='store_const', default=False)
