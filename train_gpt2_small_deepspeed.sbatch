#!/bin/bash
#SBATCH --job-name=gpt2-deepspeed
#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=g2-standard-12    
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=./logs/%x_%j.log

export TRITON_CACHE_DIR=/scratch/zl5604/triton_cache
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export LOCAL_RANK=$SLURM_LOCALID

SINGULARITY_CONTAINER=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/zl5604/project/project.ext3:rw

singularity exec --nv --overlay ${OVERLAY} ${SINGULARITY_CONTAINER} \
    /bin/bash -c "
export TRITON_CACHE_DIR=/scratch/zl5604/triton_cache;
source /ext3/env.sh;
cd /scratch/zl5604/project/;
conda activate deepspeed;
CUDA_VISIBLE_DEVICES=0 deepspeed train_gpt2_small_deepspeed.py
--micro-batch-size 8
--global-batch-size 8
--train-iters 31
--log-interval 15
--eval-interval 15
--save-interval 15
--log-throughput
--log-timers-to-tensorboard
--num-gpus 1
--num-nodes 1
--seed 42;
"