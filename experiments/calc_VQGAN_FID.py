import sys
sys.path.append('.')
from hparams import get_vqgan_hparams
from utils.log_utils import log, config_log, start_training_log
from models import VQGAN
from utils.vqgan_utils import calc_FID
from utils.log_utils import load_model


def main(H):
    vqgan = VQGAN(H).cuda()
    try:
        vqgan = load_model(vqgan, "vqgan_ema", H.load_step, H.load_dir)

    except FileNotFoundError:
        vqgan = load_model(vqgan, "vqgan", H.load_step, H.load_dir)

    fid = calc_FID(H, vqgan)
    log(fid)


if __name__ == '__main__':
    H = get_vqgan_hparams()
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for VQGAN on {H.dataset}')
    start_training_log(H)
    main(H)
