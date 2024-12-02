#!/bin/bash
#SBATCH --job-name=gpt2_pytorch
#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --output=logs/gpt2_pytorch_%j_%x.out
#SBATCH --error=logs/gpt2_pytorch_%j_%x.err
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=32G

SINGULARITY_CONTAINER=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/zl5604/project/project.ext3:rw

# Run the training script without DDP
singularity exec --nv --overlay ${OVERLAY} ${SINGULARITY_CONTAINER} \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/zl5604/project/; 
    conda activate deepspeed; \
    python train_gpt2_small.py \
    --data-path /oscar_subsets/oscar_subset_100MB_raw.jsonl \
    --model-path /gpt2_small/pytorch_model.bin \
    --vocab-file /gpt2_small/vocab.json \
    --merge-file /gpt2_small/merges.txt \
    --checkpoint-path /checkpoints/ \
    --tensorboard-logs-path /logs/ \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --train-iters 500 \
    --seed 42"
