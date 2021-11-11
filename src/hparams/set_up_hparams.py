import argparse
from .defaults.sampler_defaults import HparamsAbsorbing, HparamsAutoregressive, add_sampler_args
from .defaults.vqgan_defaults import HparamsVQGAN, add_vqgan_args
from .defaults.experiment_defaults import add_PRDC_args, add_sampler_FID_args


# args for training of all models: dataset, EMA and loading
def add_training_args(parser):
    parser.add_argument("--amp", const=True, action="store_const", default=False)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--custom_dataset_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ema_beta", type=float, default=0.995)
    parser.add_argument("--ema", const=True, action="store_const", default=False)
    parser.add_argument("--load_dir", type=str, default="test")
    parser.add_argument("--load_optim", const=True, action="store_const", default=False)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps_per_update_ema", type=int, default=10)
    parser.add_argument("--train_steps", type=int, default=100000000)


# args required for logging
def add_logging_args(parser):
    parser.add_argument("--log_dir", type=str, default="test")
    parser.add_argument("--save_individually", const=True, action="store_const", default=False)
    parser.add_argument("--steps_per_checkpoint", type=int, default=10000)
    parser.add_argument("--steps_per_display_output", type=int, default=250)
    parser.add_argument("--steps_per_eval", type=int, default=0)
    parser.add_argument("--steps_per_log", type=int, default=1)
    parser.add_argument("--steps_per_save_output", type=int, default=1000)
    parser.add_argument("--visdom_port", type=int, default=8097)
    parser.add_argument("--visdom_server", type=str)


def set_up_base_parser(parser):
    add_training_args(parser)
    add_logging_args(parser)


def apply_parser_values_to_H(H, args):
    # NOTE default args in H will be overwritten by any default parser args
    args = args.__dict__
    for arg in args:
        if args[arg] is not None:
            H[arg] = args[arg]

    return H


def get_vqgan_hparams():
    parser = argparse.ArgumentParser("Parser for setting up VQGAN training :)")
    set_up_base_parser(parser)
    add_vqgan_args(parser)
    parser_args = parser.parse_args()
    H = HparamsVQGAN(parser_args.dataset)
    H = apply_parser_values_to_H(H, parser_args)

    if not H.lr:
        H.lr = H.base_lr * H.batch_size

    return H


def get_sampler_H_from_parser(parser):
    parser_args = parser.parse_args()
    dataset = parser_args.dataset

    # has to be in this order to overwrite duplicate defaults such as batch_size and lr
    H = HparamsVQGAN(dataset)
    H.vqgan_batch_size = H.batch_size  # used for generating samples and latents

    if parser_args.sampler == "absorbing":
        H_sampler = HparamsAbsorbing(dataset)
    elif parser_args.sampler == "autoregressive":
        H_sampler = HparamsAutoregressive(dataset)
    H.update(H_sampler)  # overwrites old (vqgan) H.batch_size
    H = apply_parser_values_to_H(H, parser_args)
    return H

def set_up_sampler_parser(parser):
    set_up_base_parser(parser)
    add_vqgan_args(parser)
    add_sampler_args(parser)
    return parser


def get_sampler_hparams():
    parser = argparse.ArgumentParser("Parser for training discrete latent sampler models :)")
    set_up_sampler_parser(parser)
    H = get_sampler_H_from_parser(parser)
    return H


def get_PRDC_hparams():
    parser = argparse.ArgumentParser("Script for calculating PRDC on trained samplers")
    add_PRDC_args(parser)
    parser = set_up_sampler_parser(parser)
    H = get_sampler_H_from_parser(parser)
    return H


def get_sampler_FID_hparams():
    parser = argparse.ArgumentParser("Script for calculating FID on trained samplers")
    add_sampler_FID_args(parser)
    parser = set_up_sampler_parser(parser)
    H = get_sampler_H_from_parser(parser)
    return H
