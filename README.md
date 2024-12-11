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

The ultimate goal is to identify the most efficient framework for large-scale language model training.
## How to Run
### Set Up Environment
Ensure you have the necessary environment modules loaded (e.g., Python, CUDA, PyTorch, DeepSpeed). Install the required Python packages listed in requirements.txt if needed.
```
pip install -r requirements.txt
```
### Modify the SBATCH Script if needed
Our SBATCH scripts are configured for a multi-GPU L4 platform on NYU HPC. If your setup differs, adjust the #SBATCH options (e.g., job name, output file, number of GPUs) and model-specific or dataset-specific paths in the script as necessary.
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
