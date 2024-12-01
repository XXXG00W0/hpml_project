
---

### 3. `train_gpt2_small_distributed.sh`

```bash
#!/bin/bash

# Runs the GPT-2 small model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 # Path to model checkpoints
TENSORBOARD_LOGS_PATH=$2 # Path to TensorBoard logs
VOCAB_FILE=$3 # Path to vocab.json
MERGE_FILE=$4 # Path to merges.txt
DATA_PATH=$5 # Path to dataset (prefix)

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12 
    --hidden-size 768 
    --num-attention-heads 12 
    --seq-length 1024 
    --max-position-embeddings 1024 
)

TRAINING_ARGS=(
    --micro-batch-size 8 
    --global-batch-size 64 
    --train-iters 50000 
    --weight-decay 0.01 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
    --fp16 
    --lr 0.00015 
    --lr-decay-style cosine 
    --min-lr 1e-5 
    --lr-warmup-iters 2000 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --data-impl mmap 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 1000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
