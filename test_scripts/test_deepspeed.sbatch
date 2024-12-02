#!/bin/bash
#SBATCH --job-name=test_deepspeed
#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=test_deepspeed.log

SINGULARITY_CONTAINER=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/zl5604/project/project.ext3:rw

singularity exec --nv --overlay ${OVERLAY} ${SINGULARITY_CONTAINER} /bin/bash -c "
source /ext3/env.sh;
cd /scratch/zl5604/project/;
python test_deepspeed.py;
"