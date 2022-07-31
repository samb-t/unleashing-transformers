## Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes

This repository contains code used for the ECCV 2022 paper [Unleashing Transformers](https://arxiv.org/abs/2111.12701).

![front_page_sample](assets/samples.png)

[**Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes**](https://samb-t.github.io/unleashing-transformers)<br/>
[Sam Bond-Taylor](https://samb-t.github.io/)\*,
[Peter Hessey](https://www.durham.ac.uk/staff/peter-hessey/)\*,
[Hiroshi Sasaki](https://www.durham.ac.uk/staff/hiroshi-sasaki/),
[Toby P. Breckon](https://breckon.org/toby/),
[Chris G. Willcocks](https://cwkx.github.io/)<br/>
\* Authors contributed equally

### Abstract
>   *Whilst diffusion probabilistic models can generate high quality image content, key limitations remain in terms of both generating high-resolution imagery and their associated high computational requirements. Recent Vector-Quantized image models have overcome this limitation of image resolution but are prohibitively slow and unidirectional as they generate tokens via element-wise autoregressive sampling from the prior. By contrast, in this paper we propose a novel discrete diffusion probabilistic model prior which enables parallel prediction of Vector-Quantized tokens by using an unconstrained Transformer architecture as the backbone. During training, tokens are randomly masked in an order-agnostic manner and the Transformer learns to predict the original tokens. This parallelism of Vector-Quantized token prediction in turn facilitates unconditional generation of globally consistent high-resolution and diverse imagery at a fraction of the computational expense. In this manner, we can generate image resolutions exceeding that of the original training set samples whilst additionally provisioning per-image likelihood estimates (in a departure from generative adversarial approaches). Our approach achieves state-of-the-art results in terms of Density (LSUN Bedroom: 1.51; LSUN Churches: 1.12; FFHQ: 1.20) and Coverage (LSUN Bedroom: 0.83; LSUN Churches: 0.73; FFHQ: 0.80), and performs competitively on FID (LSUN Bedroom: 3.64; LSUN Churches: 4.07; FFHQ: 6.11) whilst offering advantages in terms of both computation and reduced training set requirements.*

![front_page_sample](assets/diagram.png)

[arXiv](https://arxiv.org/abs/2111.12701) | [BibTeX](#bibtex) | [Project Page](https://samb-t.github.io/unleashing-transformers)

### Table of Contents

- [Abstract](#abstract)
- [Table of Contents](#table-of-contents)
- [Setup](#setup)
  - [Set up conda environment](#set-up-conda-environment)
  - [Dataset Set Up](#dataset-setup)
  - [Pre-Trained Models](#pre-trained-models)
- [Commands](#commands)
  - [Set up visdom server](#set-up-visdom-server)
  - [Train a Vector-Quantized autoencoder on LSUN Churches](#train-a-vector-quantized-autoencoder-on-lsun-churches)
  - [Train an Absorbing Diffusion sampler using the above Vector-Quantized autoencoder](#train-an-absorbing-diffusion-sampler-using-the-above-vector-quantized-autoencoder)
  - [Experiments on trained Absorbing Diffusion Sampler](#experiments-on-trained-absorbing-diffusion-sampler)
- [Related Work](#related-work)
- [BibTeX](#bibtex)

## Setup

Currently, a dedicated graphics card capable of running CUDA is required to run the code used in this repository. All models used for the paper were trained on a single NVIDIA RTX 2080 Ti using CUDA version 11.1.

### Set up conda environment

To run the code in this repository we recommend you set up a virtual environment using [conda](https://docs.conda.io/en/latest/). To get set up quickly, use [miniconda](https://docs.conda.io/en/latest/miniconda.html).

Run the following command to clone this repo using [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and create and activate the conda environment `unleashing`:

```
git clone https://github.com/samb-t/unleashing-transformers.git && cd unleashing-transformers
conda create --name unleashing --file requirements.yml
conda activate unleashing  
```

You should now be able to run all commands available in the following sections.

### Dataset Setup
To configure the default paths for datasets used for training the models in this repo, simply edit `datasets.yaml` - changing the `paths` attribute of each dataset you wish to use to the path where your dataset is saved locally.


| Dataset | Official Link                                           | Academic Torrents Link |
| ------- | --------------------------------------------------------|------------------------|
| FFHQ    | [Official FFHQ](https://github.com/NVlabs/ffhq-dataset) | [Academic Torrents FFHQ](https://academictorrents.com/details/1c1e60f484e911b564de6b4d8b643e19154d5809) |
| LSUN    | [Official LSUN](https://github.com/fyu/lsun)            | [Academic Torrents LSUN](https://academictorrents.com/details/c53c374bd6de76da7fe76ed5c9e3c7c6c691c489) |

### Pre-Trained Models
Pre-trained models can be found [here](https://drive.google.com/drive/folders/1pjTYcm-2NNzuAiNEO24gSt9dXu_kGQ6b?usp=sharing). To obtain all models, download the logs folder to the root directory of this repo.

## Commands
This section contains details on the basic commands for training and calculating metrics on the Absorbing Diffusion models. All training was completed on a single NVIDIA RTX 2080 Ti and these commands presume the same level of hardware. If your GPU has less VRAM than a 2080 Ti then you may need to train using smaller batch sizes and/or smaller models than the defaults.

For a detailed list of all commands options, including altering model architecture, logging output, checkpointing frequency, etc., please add the `--help` flag to the end of your command.

All commands should be run from the head directory, i.e. the directory containing the README file. 

### Set up visdom server

Before training, you'll need to start a visdom server in order to easily view model output (loss graphs, reconstructions, etc.). To do this, run the following command:

```
visdom -p 8097
```

This starts a visdom server listening on port 8097, which is the default used by our models. If you navigate to localhost:8097 you will see be able to view the live server.

To specify a different port when training any models, use the `--visdom_port` flag.

### Train a Vector-Quantized autoencoder on LSUN Churches

The following command starts the training for a VQGAN on LSUN Churches: 
```
python3 train_vqgan.py --dataset churches --log_dir vqgan_churches --amp --ema --batch_size 4
```

As specified with the `--log_dir` flag, results will be saved to the directory `logs/vqae_churches`. This includes all logs, model checkpoints and saved outputs. The `--amp` flag enables mixed-precision training, necessary for training using a batch size of 4 (the default) on a single 2080 Ti.

### Train an Absorbing Diffusion sampler using the above Vector-Quantized autoencoder

After training the VQ model using the previous command, you'll be able to run the following commands to train a discrete diffusion prior on the latent space of the Vector-Quantized model:

```
python3 train_sampler.py --sampler absorbing --dataset churches --log_dir absorbing_churches --ae_load_dir vqgan_churches --ae_load_step 2200000 --amp --ema
```

The sampler needs to load the trained Vector-Quantized autoencoder in order to generate the latents it will use as for training (and validation). Latents are cached after the first time this is run to speed up training.

### Experiments on trained Absorbing Diffusion Sampler

This section contains simple template commands for calculating metrics and other experiments on trained samplers.

**Calculate FID**

```
python experiments/calc_FID.py --sampler absorbing --dataset churches --log_dir FID_log --ae_load_dir vqgan_churches --ae_load_step 2200000  --load_dir absorbing_churches --load_step 2000000 --ema --n_samples 50000 --temp 0.9
```

**Calculate PRDC Scores**

```
python experiments/calc_PRDC.py --sampler absorbing --dataset churches --log_dir PRDC_log --ae_load_dir vqgan_churches --ae_load_step 2200000 --load_dir absorbing_churches --load_step 2000000 --ema --n_samples 50000
```


**Calculate ELBO Estimates**

The following command fine-tunes a Vector-Quantized autoencoder to compute reconstruction likelihood, and then evaluates the ELBO of the overall model.

```
python experiments/calc_approximate_ELBO.py --sampler absorbing --dataset ffhq --log_dir nll_churches --ae_load_dir vqgan_churches --ae_load_step 2200000 --load_dir absorbing_churches --load_step 2000000 --ema --steps_per_eval 5000 --train_steps 10000
```

NOTE: the `--steps_per_eval` flag is required for this script, as a validation dataset is used. 


**Find Nearest Neighbours**

Produces a random batch of samples and finds the nearest neighbour images in the training set based on LPIPS distance.

```
python experiments/calc_nearest_neighbours.py --sampler absorbing --dataset churches --log_dir nearest_neighbours_churches --ae_load_dir vqgan_churches --ae_load_step 2200000 --load_dir absorbing_churches --load_step 2000000 --ema
```

**Generate Higher Resolution Samples**

By applying the absorbing diffusion model to various locations at once and aggregating denoising probabilities, larger samples than observed during training are able to be generated (see Figures 4 and 11).

```
python experiments/generate_big_samples.py --sampler absorbing --dataset churches --log_dir big_samples_churches --ae_load_dir vqgan_churches --ae_load_step 2200000 load_dir absorbing_churches --load_step 2000000 --ema --shape 32 16
```

Use the `--shape` flag to specify the dimensions of the latents to generate.

## Related Work

The following papers were particularly helpful when developing this work:

- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
- [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

## BibTeX

```
@inproceedings{bond2021unleashing,
  title       = {Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes},
  author      = {Sam Bond-Taylor and Peter Hessey and Hiroshi Sasaki and Toby P. Breckon and Chris G. Willcocks},
  booktitle   = {European Conference on Computer Vision (ECCV)},
  year        = {2022}
}
```
