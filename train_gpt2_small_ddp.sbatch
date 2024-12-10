#!/bin/bash
#SBATCH --job-name=gpt2_ddp
#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --output=logs/%j_%x.out
#SBATCH --partition=g2-standard-24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --mem=32G

export TRITON_CACHE_DIR=/scratch/zl5604/triton_cache

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12355
WORLD_SIZE=2
RANK=$SLURM_PROCID
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
LOCAL_RANK=$((RANK % 2))

NCCL_DEBUG=INFO
NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=^lo,docker

WANDB_MODE=online

export MASTER_ADDR MASTER_PORT WORLD_SIZE RANK OMP_NUM_THREADS NCCL_DEBUG NCCL_IB_DISABLE NCCL_SOCKET_IFNAME WANDB_MODE

SINGULARITY_CONTAINER=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/zl5604/project/project.ext3:rw

# Run the training script with DDP
singularity exec --nv --overlay ${OVERLAY} ${SINGULARITY_CONTAINER} \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/zl5604/project/; 
    conda activate deepspeed; \
    torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=12355 \
    train_gpt2_small_ddp.py \
    --micro-batch-size 8 \
    --global-batch-size 16 \
    --train-iters 31 \
    --log-interval 15 \
    --eval-interval 15 \
    --save-interval 15 \
    --log-throughput \
    --log-timers-to-tensorboard \
    --framework ddp \
    --num-gpus 2 \
    --num-nodes 1 \
    --use-pytorch-profiler \
    --profile-ranks 0\
    --seed 42"