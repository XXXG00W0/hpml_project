#!/bin/bash
#SBATCH --job-name=gpt2_dp
#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --output=logs/%j_%x.out
#SBATCH --partition=g2-standard-48
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --mem=32G

export TRITON_CACHE_DIR=/scratch/zl5604/triton_cache
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4
export LOCAL_RANK=$SLURM_LOCALID

export WANDB_MODE=online

SINGULARITY_CONTAINER=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/zl5604/project/project.ext3:rw

# Run the training script without DDP
singularity exec --nv --overlay ${OVERLAY} ${SINGULARITY_CONTAINER} \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/zl5604/project; 
    conda activate deepspeed; \
    python train_gpt2_small.py \
  --micro-batch-size 8 \
  --global-batch-size 32 \
  --train-iters 31 \
  --log-interval 15 \
  --eval-interval 15 \
  --save-interval 15 \
  --log-throughput \
  --log-timers-to-tensorboard \
  --num-gpus 4 \
  --num-nodes 1 \
  --framework dp \
  --use-pytorch-profiler \
  --seed 42"