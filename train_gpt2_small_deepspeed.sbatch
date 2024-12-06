#!/bin/bash

#SBATCH --job-name=gpt2-deepspeed          # 作业名称
#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --nodes=1                           # 1 个节点
#SBATCH --partition=g2-standard-48           # 指定 GPU 分区
#SBATCH --gres=gpu:4                        # 分配 4 张 GPU
#SBATCH --cpus-per-task=4                   # 每个任务分配 8 个 CPU 核
#SBATCH --mem=32G                           # 分配 32GB 内存
#SBATCH --time=1:00:00                     # 设置最大运行时间
#SBATCH --output=/logs/log_%j.out                 # 标准输出文件
#SBATCH --error=/logs/log_%j.err                  # 错误输出文件

# 环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRITON_CACHE_DIR=/scratch/zl5604/triton_cache

# 作业参数
GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12355
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

SINGULARITY_CONTAINER=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/zl5604/project/project.ext3:rw

# 执行训练
singularity exec --nv --overlay ${OVERLAY} ${SINGULARITY_CONTAINER} \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/zl5604/project/; 
    conda activate deepspeed; \
    deepspeed train_gpt2_small_deepspeed.py \
    --data-path oscar_subsets/oscar_subset_100MB_raw.jsonl \
    --model-path gpt2_small \
    --vocab-file gpt2_small/vocab.json \
    --merge-file gpt2_small/merges.txt \
    --micro-batch-size 8 \
    --global-batch-size 32 \
    --train-iters 1 \
    --log-interval 1 \
    --eval-interval 1 \
    --save-interval 1 \
    --log-throughput \
    --log-timers-to-tensorboard \
    --framework torch \
    --num-gpus 4 \
    --num-nodes 1 \
    --use-pytorch-profiler \
    --profile-ranks 0 \
    --seed 42
    --deepspeed"