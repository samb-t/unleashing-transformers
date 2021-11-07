# Unleashing Transformers: Parallel Token Prediction with Discrete Diffusion Probabilistic Models for Fast High-Resolution Image Generation from Vector-Quantized Codes

This is the repository containing code used for the [Unleashing Transformers paper](www.google.com) 

## README To-Do

Todo
- [ ] Tidy commands and replace with commands non-NCC users will use.
- [ ] Add nice pictures to header
- [ ] Add actual paper link once on arxiv
- [x] Give conda setup tutorial
- [ ] Add pretrained models
- [ ] Get CUDA version on NCC
- [ ] Add thanks to NCC and Durham University(?)

## Setup

### Install `conda` and `git`

**Conda**

If you already have a the conda tool available, you can skip this step.

The authors recommend setting up a virtual environment using [conda](https://docs.conda.io/en/latest/) to run the code in this repository. This will enable you to install the exact same python version and other package versions as those used to gather the experimental data included in the paper. It is possible to use other versions of python and even other virtual environment tools, but identical results cannot be guaranteed.

To get set up with conda quickly and easily, use [miniconda](https://docs.conda.io/en/latest/miniconda.html). It is available for most operating systems, is lightweight compared to the full version and requires no admin/sudo permissions. Install instructions for minoconda are available [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

**Git CLI**

Most users will have git CLI installed on their system by default. But, if not, a good setup guide for most operating systems is available [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### Set up conda environment

In any folder of your choice, run the following command:
```
git clone https://github.com/samb-t/VQGAN-EBM.git && cd VQGAN-EBM
```
This will clone this repo into your local machine and cd into it.

Next, run:
```
conda create --name <env-name> --file requirements.txt
```
replacing `<env-name>` with your desired name for the environment. 

Finally, run:

```
conda activate <env-name>
```
You should now be able to run all commands available in the following sections.

### Hardware Requirements

Currently, a dedicated graphics card capable of running CUDA is required to run the code used in this repository. All models used for the paper were trained on a single NVIDIA RTX 2080ti using CUDA version <x.x.x.x>. The largest models still took less than a week to converge. 

It is ***not recommended*** that you attempt to run models on high-resolution datasets such as LSUN and FFHQ using only a CPU, as training will be very slow.  

## Useful Commands

**NCC SLURM CLI command:**

```
srun -N 1 -c 1 --gres=gpu -p "res-gpu-small" --qos "long-high-prio" -J "<task_name_here>" -t "165:59:00" --exclude gpu1,gpu2,gpu3,gpu4 --pty

```
**Train an FFHQ VQGAN**

```
python train_vqgan.py --dataset ffhq --steps_per_log 50 --steps_per_display_output 1000 --steps_per_save_output 25000 --log_dir vqgan_sam_branch_testing --ema --steps_per_checkpoint 200000 --amp --batch_size 4 --visdom_server ncc1.clients.dur.ac.uk --visdom_port 8901 --steps_per_eval 250 --horizontal_flip --load_step 1400000 --load_optim --load_dir vqgan_sam_branch_testing
```

**Train an FFHQ Absorbing Diffusion Model**

```
python train_sampler.py --sampler absorbing --dataset ffhq --ae_load_dir <vqgan_load_dir> --ae_load_step <vqgan_load_step> --loss_type new --visdom_server ncc1.clients.dur.ac.uk --visdom_port 8904 --steps_per_log 10 --steps_per_display_output 25000 --steps_per_save_output 50000 --steps_per_checkpoint 100000 --steps_per_eval 1000 --batch_size 512 
```

**Calc FID on FFHQ Absorbing Diffusion Model**

```
python calc_FID.py --sampler absorbing --dataset ffhq --ema --visdom_server ncc1.clients.dur.ac.uk --ae_load_dir vqgan_ffhq_with_hflip --ae_load_step 1400000  --load_dir absorbing_ffhq_80M_new_loss_hflip_with_trained_vqgan --load_step 700000 --sample_type v2 --stepping magic-256 --n_samples 10000 --bert_n_emb 512 --bert_n_head 8 --bert_n_layers 24
```


## Features To be Added
This is a list of features that we hope to add in the near future:
- [ ] Tidier Code
- [ ] Centralised checkpointing
- [ ] Easier integration of new datasets