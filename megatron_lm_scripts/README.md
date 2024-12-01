# GPT-2 Small Training Guide

## Table of Contents
- [1. Training Setup](#1-training-setup)
- [2. Configurations](#2-configurations)
- [3. Running the Training Script](#3-running-the-training-script)

---

## 1. Training Setup
To train the GPT-2 small model on a distributed system:

1. Use the provided Docker/Singularity container to ensure dependencies are met.
2. Adjust paths for dataset, vocabulary, and model checkpoints in the script `train_gpt2_small_distributed.sh`.
3. Configure hardware resources (e.g., GPUs per node, total nodes) based on your cluster.

---

## 2. Configurations
This setup trains the GPT-2 small model with:
- 12 layers, hidden size of 768, 12 attention heads
- Sequence length of 1024
- Tensor model parallelism: 1
- Pipeline model parallelism: 1

See `gpt2_config.yaml` for more details.

---

## 3. Running the Training Script
Use the following command to start training:
```bash
bash train_gpt2_small_distributed.sh <path_to_checkpoint> <path_to_tensorboard_logs> <path_to_vocab> <path_to_merges> <path_to_data>
