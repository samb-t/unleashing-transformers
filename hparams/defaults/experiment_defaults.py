def add_sampler_FID_args(parser):
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--latents_path", type=str)


def add_vqgan_FID_args(parser):
    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="Number of reconstructions to use for calculating FID"
    )


def add_PRDC_args(parser):
    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="Number of fake images to generate and real images to use for metric calculation"
    )
    parser.add_argument("--real_feats", type=str, help="Name of (pkl) file containing real features")
    parser.add_argument(
        "--fake_feats",
        type=str,
        help="Name of (pkl) file containing fake features, stored in src/_pkl_files/"
    )
    parser.add_argument(
        "--fake_images_path",
        type=str,
        help="Path to folder containing sampled images, if not provided, will attempt to generate images instead"
    )


def add_big_sample_args(parser):
    parser.add_argument("--shape", type=int, nargs=2, help="Shape of latents to generate. Pass as two seperate integers"
                        ", in the form H W", required=True)
