# Benchmarking Distributed Machine Learning Frameworks on GPT-2
## Team Members
**Ziyi Liang (zl5604),**
**Yihua Yang (yy5028)**
## Course
**ECE 9143 High Performance Machine Learning, Fall 2024**
## Objective
This project aims to benchmark the performance of four distributed training frameworks—PyTorch Data-Parallel (DP), PyTorch Distributed Data-Parallel (DDP), DeepSpeed stage 2, and DeepSpeed stage 3—by training a GPT-2 Small model (~124 million parameters) using a subset of the OSCAR 23.01 dataset (~100 MB). Key performance metrics include:

* Communication overhead
* Throughput
* Training time
* GPU/CPU utilization
* Memory consumption
* Scalability

The ultimate goal is to identify the most efficient framework for large language model training.

## Project Milestones and Completion Status
## Milestone 1: Setting up the Training Environment and Dataset: (Completed)
- Subset the original OSCAR dataset and tokenize it using the GPT-2 tokenizer.
- Configure distributed frameworks (Horovod, PyTorch DDP, and DeepSpeed) on the NYU HPC.
## Training Implementation: (Completed)
- Establish a single-GPU training baseline for comparison with distributed training.
- Conduct distributed training experiments using consistent configurations across frameworks.
## Profiling and Resource Monitoring: (Completed)
- Measure training time and evaluate speedup compared to the single-GPU baseline.
- Monitor memory consumption and GPU utilization to identify potential bottlenecks.
## Comparative Analysis: (Completed)
- Compare distributed frameworks' performance in training time, memory consumption, GPU utilization, and speedup.
- Analyze accuracy, evaluation loss, and perplexity (PPL) to assess the impact of optimizations by different frameworks.

## Repository Structure
The repository is organized as follows:
```
documents/                          # Project data visualization
logs/                               # Profiling logs and outputs
download_scripts/                   # Scripts for downloading and preprocessing the dataset, and downloading the GPT-2 model
megatron_lm_scripts/                # Training scripts for GPT-2 using Megatron-LM
oscar_subsets/                      # Subset of the OSCAR dataset for training
test_scripts/                       # Test scripts for benchmarking distributed training frameworks
wandb/                              # Weights and Biases logs
timers.py                           # Timer utility functions from Megatron-LM
README.md                           # Project overview and instructions
requirements.txt                    # Required Python packages
train_gpt2_small.py                 # Training script for GPT-2 Small with PyTorch and DP
train_gpt2_small_ddp.py             # Training script for GPT-2 Small with DDP
train_gpt2_small_deepspeed.py       # Training script for GPT-2 Small with DeepSpeed
train_gpt2_small.sbatch             # SLURM script for training GPT-2 Small
train_gpt2_small_dp.sbatch          # SLURM script for training GPT-2 Small with DP
train_gpt2_small_ddp.sbatch         # SLURM script for training GPT-2 Small with DDP
train_gpt2_small_deepspeed.sbatch   # SLURM script for training GPT-2 Small with DeepSpeed
```

## Code Structure
The codebase is structured as follows:

1. Imports
Essential libraries for machine learning, such as torch, transformers, and wandb.
Utility libraries for GPU monitoring, argument parsing, data handling, and logging.
Custom modules like Timers and utility functions.
2. Argument Parsing
parse_args() defines a comprehensive list of command-line arguments for:
Model and data paths: Paths to model, vocabulary, merge files, checkpoints, etc.
Model parameters: Configuration like sequence length, hidden size, number of layers.
Training parameters: Batch sizes, learning rate, and gradient clipping.
Logging and checkpointing: Logging intervals, TensorBoard paths, and WandB settings.
Distributed training: Number of GPUs, nodes, and distributed framework configurations.
3. Directory Management
check_directory(args) ensures the required directories (e.g., for checkpoints and TensorBoard logs) exist and creates them if necessary.
4. Dataset Preparation
TextDataset class: Reads and tokenizes the dataset into fixed-size blocks for training.
create_datasets: Splits the dataset into training and validation sets.
create_train_val_dataloader: Creates PyTorch DataLoaders for both sets with batching and multi-threaded data loading.
5. GPU Monitoring
get_gpu_info: Collects and displays GPU usage statistics such as memory usage and temperature.
6. Evaluation
evaluate: Computes average loss and perplexity on the validation set while managing GPU memory.
7. Learning Rate Scheduler
create_warmup_cosine_schedule: Combines a warmup phase with a cosine decay for learning rate adjustment.
8. Model Setup
setup_no_distributed: Moves the model to GPU and sets up optimizers, schedulers, and mixed precision (FP16) support.
9. Training Step
train_step_torch: Handles a single training iteration, including loss computation, gradient scaling, and memory management.
10. Main Training Loop
main: Orchestrates the training process:
Parses arguments and initializes logging (e.g., TensorBoard, WandB).
Loads datasets and model configurations.
Executes the training loop, which includes:
Training steps with profiling and logging.
Periodic validation and checkpoint saving.
Conducts a final evaluation and saves the trained model.
11. Profiling and Logging
Integrated with PyTorch Profiler and TensorBoard for detailed performance insights.
Logs throughput, memory usage, loss, and other metrics to TensorBoard and WandB.
12. Finalization
Saves the final model checkpoint and cleans up resources (e.g., closing TensorBoard writers, WandB sessions).


## How to Run
### Set Up Environment
Ensure you have the necessary environment modules loaded (e.g., Python, CUDA, PyTorch, DeepSpeed). Install the required Python packages listed in requirements.txt if needed.
```
pip install -r requirements.txt
```

### Modify the SBATCH Script if needed
Our SBATCH scripts are configured for a multi-GPU L4 platform on NYU HPC. If your setup differs, adjust the #SBATCH options (e.g., job name, output file, number of GPUs) and model-specific or dataset-specific paths in the script as necessary.

Some key parameters to modify include:
```
$WORLD_SIZE: Number of GPUs to use for training
CUDA_VISIBLE_DEVICES=0,1,2,3: List of GPU IDs to use for training
--num-gpus: Number of GPUs to use for training
--framework: Distributed training framework (dp, ddp, deepspeed, none)
--nproc_per_node=2 : Number of GPUs per node

### Submit the SBATCH Script
Use the sbatch command to submit the job to the SLURM queue. For example:
```
sbatch train_gpt2_small_ddp.sbatch
```
### Outputs and Logs
After the job completes, logs will be saved to the specified output file in the SBATCH script. Profiling data (e.g., GPU utilization, communication times) will be stored in the ```logs/``` directory.

## Results
### Scalability Performance Relative to Single GPU
| # of GPU | PyTorch Baseline | DDP   | DeepSpeed Stage 2 | DeepSpeed Stage 3 |
|----------|------------------|-------|-------------------|-------------------|
| 1        | 1.0              | 1.0   | 1.0               | 1.0               |
| 2        | 2.246            | 1.910 | 1.466             | 1.575             |
| 4        | 3.881            | 3.653 | 3.76              | 2.472             |

### Throughput
![image](https://github.com/user-attachments/assets/1a521ff6-a02d-4c63-b1fe-5b0cdd9fe658)

### Total Time
![image](https://github.com/user-attachments/assets/e0f80cb3-12e6-446e-a031-55a835a48abc)

### Total Communication Time
![image](https://github.com/user-attachments/assets/7a82316a-f1ae-4d46-8ccb-f44bc9110f4b)

## Summary
We evaluated the performance of PyTorch Baseline, Distributed Data Parallel (DDP), and DeepSpeed (Stages 2 and 3) on a GPT-2 Small model across 1, 2, and 4 GPUs. The key findings are:

### Aggregated Throughput:

PyTorch Baseline achieved the highest throughput across all GPU configurations, showing excellent scaling performance.
DDP demonstrated a slightly lower throughput compared to the baseline but maintained near-linear scaling with increased GPUs.
DeepSpeed Stage 2 closely matched the performance of PyTorch Baseline at 4 GPUs, outperforming DDP in scaling efficiency.
DeepSpeed Stage 3 lagged in throughput, particularly at 4 GPUs, indicating potential overheads from its more aggressive memory optimizations.
### Total Time Spent (CPU vs. CUDA):

DDP and DeepSpeed both exhibited balanced device and CPU time utilization.
PyTorch Baseline showed higher CPU utilization due to its synchronization and data management overheads.
DeepSpeed Stage 3 displayed significant reduction in memory and compute overhead, particularly for larger GPU configurations.
### Distributed Communication:

DDP and DeepSpeed demonstrated efficient use of NCCL for distributed communication, with DDP being slightly faster in small-scale GPU configurations.
DeepSpeed's memory optimization strategies (Stage 3) reduced communication overhead but impacted overall throughput at higher GPU counts.
These results provide insight into the trade-offs between memory optimization, throughput, and communication efficiency across distributed training frameworks. For large-scale training tasks requiring memory efficiency, DeepSpeed Stage 3 offers a viable solution, while PyTorch Baseline remains ideal for maximizing raw throughput.
