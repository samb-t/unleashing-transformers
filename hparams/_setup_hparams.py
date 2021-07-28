# Resources for managing hyperparameters for both stage 1 and 2 of training 
# loads all params that are shared by both stages (e.g. batch size)
import argparse
import deepspeed
from vqgan_params import add_vqgan_args


def set_up_H(H, args):

    # replace defaults with user-specified arguments

    # no fix required here - just do not set defaults arg parser and in Hparams classes
    # Hparams classes defaults should ONLY be used for architectural and training param args
    # wheras parser defaults should be use for logging stuff only
    args = args.__dict__
    for arg in args:
        if args[arg] != None:
            H[arg] = args[arg]
    

# args for training of all models: dataset, EMA and loading
def add_training_args(parser):
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--ema', const=True, action='store_const', default=False)
    parser.add_argument('--ema_beta', type=float, default=0.995)
    parser.add_argument('--steps_per_update_ema', type=int, default=10)
    parser.add_argument('--amp', const=True, action='store_const', default=False)
    parser.add_argument('--train_steps', type=int, default=100000000)
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--load_dir', type=str, default='test')
    parser.add_argument('--load_optim', const=True, action='store_const', default=False)


# args required for logging
def add_logging_args(parser):
    parser.add_argument('--log_dir', type=str, default='test')
    parser.add_argument('--visdom_port', type=int, default=8097)
    parser.add_argument('--ncc', const=True, action='store_const', default=False)
    parser.add_argument('--steps_per_log', type=int, default=1)
    parser.add_argument('--steps_per_checkpoint', type=int, default=1000)
    parser.add_argument('--steps_per_display_output', type=int, default=50)
    parser.add_argument('--steps_per_save_output', type=int, default=100)
    parser.add_argument('--save_individually', const=True, action='store_const', default=False)


def add_deepspeed_args(parser):
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=0)



def setup_base_parser(parser):
    add_training_args(parser)
    add_logging_args(parser)
    add_deepspeed_args(parser)


def get_vqgan_hparams():
    parser = argparse.ArgumentParser('Parser for setting up VQGAN training :)')
    setup_base_parser(parser)
    
    '''
    - set up parser args
    - parse them
    - apply VQGAN defaults
    - set learning rate
        - Check if a fixed lr has been passed in
    '''
    H.set_vqgan_lr()

def get_diffusion_decoder_hparams():
    ...


def get_sampler_hparams():
    ...
    # calls to sampler hparam py file




'''
- Can't use seperate get_ functions for samplers unfortunately, will set up loggin seperately
- Will want to set up deepspeed in this section I believe
    - Include in collective args function? (i.e. batch size, learning rate, logging args etc.)
    - batch_size may not be communal to both, as will need seperate batch size for ae when producing samples while training sampler
    - can specify ae_batch_size for samplers I suppose
'''