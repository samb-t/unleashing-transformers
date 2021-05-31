class Hparams():
    def __init__(self, dataset):
        self.dataset = dataset
        if self.dataset == 'mnist':
            self.batch_size = 128
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
            self.perceptual_weight = 0.0
            self.disc_start_step = 2001
            self.vq_base_lr = 4.5e-6

            #ebm only params
            self.buffer_size = 10000
            self.sampling_steps = 50
            self.warmup_iters = 2000
            self.ebm_lr = 1e-4
            self.l2_coef = 1
            self.latent_shape = [1, 8, 8]
        
        elif self.dataset == 'cifar10':
            self.batch_size = 128
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
            self.perceptual_weight = 1.0
            self.disc_start_step = 10001
            self.vq_base_lr = 4.5e-6

            #ebm only params
            self.buffer_size = 10000
            self.sampling_steps = 50
            self.warmup_iters = 2000
            self.ebm_lr = 1e-5
            self.l2_coef = 0
            self.latent_shape = [1, 8, 8]

        elif self.dataset == 'flowers':
            self.batch_size = 128
            self.img_size = 32
            self.n_channels = 3
            self.nf = 64
            self.ndf = 32
            self.ch_mult = [1,1,2]
            self.attn_resolutions = [8]
            self.res_blocks = 1
            self.disc_layers = 1
            self.codebook_size = 128
            self.emb_dim = 128
            self.perceptual_weight = 1.0
            self.disc_start_step = 10001
            self.vq_base_lr = 4.5e-6

            #ebm only params
            self.buffer_size = 1000
            self.sampling_steps = 50
            self.warmup_iters = 2000
            self.ebm_lr = 1e-4
            self.l2_coef = 0
            self.latent_shape = [1, 8, 8]

        elif self.dataset == 'celeba':
            self.batch_size = 3
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
            self.perceptual_weight = 1.0
            self.disc_start_step = 30001
            self.vq_base_lr = 4.5e-6

            # ebm only params
            self.buffer_size = 10000 # might be way too large
            self.sampling_steps = 50
            self.warmup_iters = 2000
            self.ebm_lr = 1e-4
            self.l2_coef = 0
            self.latent_shape = [1, 16, 16]

        else:
            raise KeyError(f'Unknown dataset: {self.dataset}')

    def get_vqgan_param_dict(self):
        return dict(
            dataset = self.dataset,
            batch_size = self.batch_size,
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

    
    def get_ebm_param_dict(self):
        return dict(
            dataset = self.dataset,
            batch_size = self.batch_size,
            img_size = self.img_size,
            n_channels = self.n_channels,
            codebook_size = self.codebook_size,
            emb_dim = self.emb_dim,
            buffer_size = self.buffer_size,
            sampling_steps = self.sampling_steps,
            warmup_iters = self.warmup_iters,
            ebm_lr = self.ebm_lr,
            l2_coef = self.l2_coef,
            latent_shape = self.latent_shape
        )