import argparse

from torch.utils.data import dataset

class Hparams(dict):
    def __init__(self, dataset):
        self.dataset = dataset
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

            # vqgan training defaults
            self.vqgan_batch_size = 128
            self.perceptual_weight = 0.0
            self.disc_start_step = 2001
            self.vq_base_lr = 4.5e-6

            # bert architcture defaults
            self.latent_shape = [1, 8, 8]
            self.block_size = 512
            self.bert_n_layers = 16
            self.bert_n_head = 8
            self.bert_n_emb = 512

            # bert training defaults
            self.bert_batch_size = 32
            self.bert_lr = 1e-4
            
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

            # vqgan training defaults
            self.vqgan_batch_size = 128
            self.perceptual_weight = 1.0
            self.disc_start_step = 10001
            self.vq_base_lr = 4.5e-6

            # bert architcture defaults
            self.latent_shape = [1, 8, 8]
            self.block_size = 256
            self.bert_n_layers = 8
            self.bert_n_head = 8
            self.bert_n_emb = 256

            # bert training defaults
            self.bert_batch_size = 32
            self.bert_lr = 1e-4


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

            # vqgan training defaults
            self.vqgan_batch_size = 128
            self.perceptual_weight = 1.0
            self.disc_start_step = 10001
            self.vq_base_lr = 4.5e-6

            # bert architcture defaults
            self.latent_shape = [1, 8, 8]
            self.block_size = 512
            self.bert_n_layers = 16
            self.bert_n_head = 8
            self.bert_n_emb = 512

            # bert training defaults
            self.bert_batch_size = 32
            self.bert_lr = 1e-4


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

            # vqgan training defaults
            self.vqgan_batch_size = 3
            self.perceptual_weight = 1.0
            self.disc_start_step = 30001
            self.vq_base_lr = 4.5e-6 

            # bert architcture defaults
            self.latent_shape = [1, 8, 8]
            self.block_size = 512
            self.bert_n_layers = 16
            self.bert_n_head = 8
            self.bert_n_emb = 512

            # bert training defaults
            self.bert_batch_size = 32
            self.bert_lr = 1e-4

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

            # vqgan training defaults
            self.vqgan_batch_size = 3
            self.perceptual_weight = 1.0
            self.disc_start_step = 30001
            self.vq_base_lr = 4.5e-6 

            # bert architcture defaults
            self.latent_shape = [1, 16, 16]
            self.block_size = 256
            self.bert_n_layers = 8
            self.bert_n_head = 8
            self.bert_n_emb = 256

            # bert training defaults
            self.bert_batch_size = 32
            self.bert_lr = 1e-4

        elif self.dataset == None:
            raise KeyError('Please specify a dataset using the -d flag')
        else:
            raise KeyError(f'Unknown dataset: {self.dataset}')


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
            perceptual_weight=self.perceptual_weight,
            disc_start_step=self.disc_start_step,
            codebook_size = self.codebook_size,
            emb_dim = self.emb_dim
        )


    def get_bert_param_dict(self):
        return dict(
            dataset = self.dataset,
            batch_size = self.bert_batch_size,
            ebm_lr = self.bert_lr,
            img_size = self.img_size,
            n_channels = self.n_channels,
            codebook_size = self.codebook_size,
            emb_dim = self.emb_dim,
            latent_shape = self.latent_shape,
            block_size = self.block_size,
            bert_n_layers = self.bert_n_layers,
            bert_n_head = self.bert_n_head,
            bert_n_emb = self.bert_n_emb,
        )


    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None


    def __setattr__(self, attr, value):
        self[attr] = value

    def __str__(self):
        return f'H object of dataset: {self.dataset}'


def get_hparams():
    parser = argparse.ArgumentParser(description='Arguments for training a VQGAN')
    
    # required args
    parser.add_argument('-d', dest='dataset', type=str)
    parser.add_argument('--ae_load_step', dest='ae_load_step', type=int) # ebm only

    # training loop control args

    parser.add_argument('--train_steps', dest='train_steps', type=int, default=1000000)
    parser.add_argument('--load_step', dest='load_step', type=int, default=0)
    parser.add_argument('--load_optim', dest='load_optim', const=True, action='store_const', default=False)

    # logging args
    
    parser.add_argument('--log_dir', dest='log_dir', type=str, default='test')
    parser.add_argument('--visdom_port', dest='visdom_port', type=int, default=8097)
    parser.add_argument('--ncc', dest='ncc', const=True, action='store_const', default=False)

    ## vqgan
    parser.add_argument('--steps_per_log', dest='steps_per_log', type=int, default=1)
    parser.add_argument('--steps_per_display_recons', dest='steps_per_display_recons', type=int, default=5)
    parser.add_argument('--steps_per_save_recons', dest='steps_per_save_recons', type=int, default=1000)
    parser.add_argument('--steps_per_vqgan_checkpoint', dest='steps_per_vqgan_checkpoint', type=int, default=10000)

    ## bert
    parser.add_argument('--steps_per_display_samples', dest='steps_per_display_samples', type=int, default=50)
    parser.add_argument('--steps_per_save_samples', dest='steps_per_save_samples', type=int, default=100)
    parser.add_argument('--steps_per_bert_checkpoint', dest='steps_per_ebm_checkpoint', type=int, default=1000)

    # vqgan architecture args
    parser.add_argument('--img_size', dest='img_size', type=int)
    parser.add_argument('--n_channels', dest='n_channels', type=int)
    parser.add_argument('--nf', dest='nf', type=int)
    parser.add_argument('--ndf', dest='ndf', type=int)
    parser.add_argument('--ch_mult', dest='ch_mult', nargs='+', type=int)
    parser.add_argument('--attn_resolutions', dest='attn_resolutions', nargs='+', type=int)
    parser.add_argument('--res_blocks', dest='res_blocks', type=int)
    parser.add_argument('--disc_layers', dest='disc_layers', type=int)
    parser.add_argument('--codebook_size', dest='codebook_size', type=int)
    parser.add_argument('--emb_dim', dest='emb_dim', type=int)
    
    # vqgan training args
    parser.add_argument('--vqgan_bs', dest='vqgan_batch_size', type=int)
    parser.add_argument('--perceptual_weight', dest='perceptual_weight', type=int)
    parser.add_argument('--disc_start_step', dest='disc_start_step', type=int)
    parser.add_argument('--vq_base_lr', dest='vq_base_lr', type=float)

    # bert architecture args
    parser.add_argument('--latent_shape', dest='latent_shape', nargs='+', type=int)
    parser.add_argument('--block_size', dest='block_size', type=int)
    parser.add_argument('--bert_n_layers', dest='bert_n_layers', type=int)
    parser.add_argument('--bert_n_head', dest='bert_n_head', type=int)
    parser.add_argument('--bert_n_emb', dest='bert_n_emb', type=int)

    # bert training args
    parser.add_argument('--bert_batch_size', dest='ebm_batch_size', type=int)
    parser.add_argument('--bert_lr', dest='ebm_lr', type=float)

    args = parser.parse_args().__dict__
    dataset = args['dataset']
    H = Hparams(dataset)
    for arg in args:
        if args[arg] != None:
            H[arg] = args[arg]
        # print(f'arg: {arg}, H val: {H[arg]}')
    H.set_vqgan_lr()
    return H

''' args for bert:

-bert_batch_size
- block_size
- bert_n_layers
- bert_n_head
- bert_n_emb

'''