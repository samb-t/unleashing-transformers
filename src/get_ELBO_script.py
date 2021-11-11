from hparams import get_sampler_hparams
from utils.log_utils import load_stats, config_log, log


def main(H):
    stats = load_stats(H, 500000)
    print(len(stats))


if __name__ == "__main__":
    H = get_sampler_hparams()
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    main(H)
