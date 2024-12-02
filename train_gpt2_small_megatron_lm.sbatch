#!/bin/bash

#SBATCH --job-name=gpt2-small-train          # 作业名称
#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --nodes=1                           # 1 个节点
#SBATCH --partition=c12m85-a100-1           # 指定 GPU 分区
#SBATCH --gres=gpu:4                        # 分配 4 张 GPU
#SBATCH --cpus-per-task=8                   # 每个任务分配 8 个 CPU 核
#SBATCH --mem=32G                           # 分配 32GB 内存
#SBATCH --time=1:00:00                     # 设置最大运行时间
#SBATCH --output=/logs/log_%j.out                 # 标准输出文件
#SBATCH --error=/logs/log_%j.err                  # 错误输出文件

# 加载必要模块
module load singularity
module load cuda/11.8  # 确保版本与运行环境匹配

# 环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 作业参数
GPUS_PER_NODE=4
MASTER_ADDR=$(hostname)
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

# 用户需要提供以下路径
CHECKPOINT_PATH="/checkpoints/"       # 模型检查点保存路径
TENSORBOARD_LOGS_PATH="/logs/"        # TensorBoard 日志路径
VOCAB_FILE="../gpt2_small/vocab.json"             # 词汇表文件
MERGE_FILE="../gpt2_small/merges.txt"             # BPE 合并规则文件
DATA_PATH="gpt2-small"             # 数据集路径（前缀）

# 分布式训练参数
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# 模型参数
GPT_MODEL_ARGS=(
    --num-layers 12 
    --hidden-size 768 
    --num-attention-heads 12 
    --seq-length 1024 
    --max-position-embeddings 1024 
)

# 训练参数
TRAINING_ARGS=(
    --micro-batch-size 8 
    --global-batch-size 32
    --train-iters 500 # shorted to 500 for benchmarking 
    --weight-decay 0.01 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
    --fp16 
    --lr 0.00015 
    --lr-decay-style cosine 
    --min-lr 1e-5 
    --lr-warmup-iters 2000 
    --use-pytorch-profiler
    # dynamic get list of ranks from GPUS_PER_NODE
    --profile-ranks $(seq -s, 0 $(($GPUS_PER_NODE - 1)))
    # --empty-unused-memory-level 1    
)

INITIALIATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

# 数据和日志参数
DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --data-impl mmap 
    --split 949,50,1
)

# 评估和日志记录参数
EVAL_AND_LOGGING_ARGS=(
    --log-straggler
    --log-interval 100
    --save-interval 1000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --log-throughput
    --timing-log-level 2
    --timing-log-option all
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --wandb-project megatron-gpt2-benchmark
    --wandb-save-dir ./wandb

)

SINGULARITY_CONTAINER=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/zl5604/project/project.ext3:rw

# 执行训练
singularity exec --nv --overlay ${OVERLAY} ${SINGULARITY_CONTAINER} \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/zl5604/project/; 
    conda activate megatron-lm \
    torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIATION_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}"
