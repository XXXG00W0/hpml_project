#!/bin/bash

#SBATCH --job-name=test_horovod
#SBATCH --account=ece_gy_9143-2024fa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
#SBATCH --output=test_horovod_$TIMESTAMP.log

SINGULARITY_CONTAINER=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/zl5604/project/project.ext3:rw

singularity exec --nv --overlay ${OVERLAY} ${SINGULARITY_CONTAINER} /bin/bash -c "
source /ext3/env.sh;
conda activate horovod-env-py312;
cd /scratch/zl5604/project/;
python test_horovod.py;
conda deactivate;
"