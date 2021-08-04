# Resources for managing hyperparameters for both stage 1 and 2 of training 
# loads all params that are shared by both stages (e.g. batch size)
import argparse
import deepspeed
from .defaults import \
    HparamsVQGAN, HparamsAbsorbing, HparamsAutoregressive, \
    add_sampler_args, add_vqgan_args


# args for training of all models: dataset, EMA and loading
def add_training_args(parser):
    parser.add_argument('--amp', const=True, action='store_const', default=False)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ema_beta', type=float, default=0.995)
    parser.add_argument('--ema', const=True, action='store_const', default=False)
    parser.add_argument('--load_dir', type=str, default='test')
    parser.add_argument('--load_optim', const=True, action='store_const', default=False)
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--steps_per_update_ema', type=int, default=10)
    parser.add_argument('--train_steps', type=int, default=100000000)


# args required for logging
def add_logging_args(parser):
    parser.add_argument('--log_dir', type=str, default='test')
    parser.add_argument('--ncc', const=True, action='store_const', default=False)
    parser.add_argument('--save_individually', const=True, action='store_const', default=False)
    parser.add_argument('--steps_per_checkpoint', type=int, default=10000)
    parser.add_argument('--steps_per_display_output', type=int, default=250)
    parser.add_argument('--steps_per_eval', type=int, default=0)
    parser.add_argument('--steps_per_log', type=int, default=1)
    parser.add_argument('--steps_per_save_output', type=int, default=1000)
    parser.add_argument('--visdom_port', type=int, default=8097)


def add_deepspeed_args(parser):
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=0)


def set_up_base_parser(parser):
    add_training_args(parser)
    add_logging_args(parser)
    add_deepspeed_args(parser)


def set_up_H(H, args):
    # DEFAULT ARGS IN H WILL BE OVERWRITTEN BY ANY DEFAULT PARSER ARGS
    args = args.__dict__
    for arg in args:
        if args[arg] != None:
            H[arg] = args[arg]

    return H

def get_vqgan_hparams():
    parser = argparse.ArgumentParser('Parser for setting up VQGAN training :)')
    set_up_base_parser(parser)
    add_vqgan_args(parser)
    parser_args = parser.parse_args()
    H = HparamsVQGAN(parser_args.dataset)
    H = set_up_H(H, parser_args)
    
    if not H.lr:
        H.lr = H.base_lr * H.batch_size
    
    return H
    
def get_diffusion_decoder_hparams():
    ...


def get_sampler_hparams():
    parser = argparse.ArgumentParser('Parser for training discrete latent sampler models :)')
    set_up_base_parser(parser)
    add_vqgan_args(parser) # necessary for loading decoder/generator for displaying samples
    add_sampler_args(parser)
    parser_args = parser.parse_args()
    dataset = parser_args.dataset
    
    # has to be in this order to overwrite duplicate defaults such as batch_size and lr
    H = HparamsVQGAN(dataset)
    H.vqgan_batch_size = H.batch_size # used for generating samples and latents
        
    if parser_args.sampler  == 'absorbing':
        H_sampler = HparamsAbsorbing(dataset)
    elif parser_args.sampler  == 'bert':
        H_sampler = HparamsAutoregressive(dataset)
    elif parser_args.sampler == 'autoregressive':
        H_sampler = HparamsAutoregressive(dataset)
    else:
        # other models go here
        ... 
    H.update(H_sampler) # overwrites old (vqgan) H.batch_size
    H = set_up_H(H, parser_args)
    return H