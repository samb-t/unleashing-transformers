# Unleashing Transformers: Parallel Token Prediction with Discrete Diffusion Probabilistic Models for Fast High-Resolution Image Generation from Vector-Quantized Codes

This is the repository containing code used for the [Unleashing Transformers paper](www.google.com) 

## Useful Commands
An example command for running 80M FFHQ Model:

**NCC SLURM:**

```
srun -N 1 -c 1 --gres=gpu -p "res-gpu-small" --qos "long-high-prio" -J "<task_name_here>" -t "165:59:00" --exclude gpu1,gpu2,gpu3,gpu4 --pty
```

**Train an FFHQ VQGAN**
```
python train_vqgan.py --dataset ffhq --steps_per_log 50 --steps_per_display_output 1000 --steps_per_save_output 25000 --log_dir vqgan_sam_branch_testing --ema --steps_per_checkpoint 200000 --amp --batch_size 4 --ncc --visdom_port 8901 --steps_per_eval 250 --horizontal_flip --load_step 1400000 --load_optim --load_dir vqgan_sam_branch_testing
```



## Features To be Added
- [ ] Tidier Code
- [ ] Centralised checkpointing