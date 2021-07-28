from hparams import get_vqgan_hparams
from utils import *

if __name__=='__main__':
    H = get_vqgan_hparams()
    vis = setup_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.model}')   
    start_training_log(H)
