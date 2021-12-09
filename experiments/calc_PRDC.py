import sys
sys.path.append('.')
import torch
import time
from hparams import get_PRDC_hparams
from torch_fidelity.utils import create_feature_extractor
from tqdm import tqdm
from utils.data_utils import BigDataset, NoClassDataset, get_datasets
from utils.log_utils import log, config_log, start_training_log
from utils.experiment_utils import generate_samples
from prdc import compute_prdc


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class Distance:
    def __init__(self):
        super().__init__()
        self.feat_extractor = create_feature_extractor('inception-v3-compat', ['2048']).cuda()
        self.distance_metric = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def extract_feats(self, in0):
        feats = self.feat_extractor.forward(in0)[0]
        return feats

    def compare_images(self, feats0, feats1):
        return self.distance_metric(feats0, feats1)


def get_feats_from_loader(data_loader):
    distance_fn = Distance()
    features = []
    for batch in tqdm(data_loader):
        batch = batch.cuda()
        feats = distance_fn.extract_feats(batch)
        features.append(feats.cpu())

    return features


def main(H):
    # get features from original dataset
    if not H.real_feats:
        log(f"Generating real features for {H.dataset}")
        real_dataset, _ = get_datasets(H.dataset, H.img_size, custom_dataset_path=H.custom_dataset_path)
        real_dataset = NoClassDataset(real_dataset, H.n_samples)  # n_images defaults to None
        real_data_loader = torch.utils.data.DataLoader(real_dataset, batch_size=H.batch_size)
        real_features = get_feats_from_loader(real_data_loader)
        timestamp = int(time.time())
        torch.save(real_features, f"src/_pkl_files/{H.dataset}_real_features_{timestamp}.pkl")

    else:
        log(f"Loading real features from src/_pkl_files/{H.real_feats}")
        real_features = torch.load(f"src/_pkl_files/{H.real_feats}")

    # get features from model-generated samples
    if not H.fake_feats:
        if not H.fake_images_path:
            log(f"Generating {H.n_samples} samples")
            generate_samples(H)
            fake_images_path = f"logs/{H.log_dir}/images/"
        else:
            fake_images_path = H.fake_images_path

        fake_dataset = BigDataset(fake_images_path)
        fake_data_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=H.batch_size)
        log("Generating fake samples features")
        fake_features = get_feats_from_loader(fake_data_loader)
        timestamp = int(time.time())
        torch.save(fake_features, f"src/_pkl_files/{H.dataset}_fake_features_{timestamp}.pkl")

    else:
        log(f"Loading fake features from src/_pkl_files/{H.fake_feats}")
        fake_features = torch.load(f"src/_pkl_files/{H.fake_feats}")

    real_features = torch.cat(real_features, dim=0)[:H.n_samples].numpy()
    fake_features = torch.cat(fake_features, dim=0)[:H.n_samples].numpy()

    log("Computing PRDC metrics...")
    metrics = compute_prdc(
        real_features=real_features,
        fake_features=fake_features,
        nearest_k=3
    )
    log(metrics)


if __name__ == '__main__':
    H = get_PRDC_hparams()
    config_log(H.log_dir)
    log('---------------------------------')
    if H.load_step > 0:
        log(f'Running PRDC calculations for sampler in {H.load_dir} at step {H.load_step}')
        start_training_log(H)
        main(H)
    else:
        raise ValueError("No value provided for --load_step, cannot calculate FID for new model")
