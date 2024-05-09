#!/bin/bash
#SBATCH --account=project_462000189
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=../stcat_output/OUTPUT.out

module use /appl/local/csc/soft/ai/modulefiles/
module load pytorch

export PYTHONUSERBASE=/scratch/project_462000189/hans/lib/python3.10/site-packages

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/project_462000189/hans/ffmpeg/lib
export PATH=$PATH:/scratch/project_462000189/hans/ffmpeg/bin
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/scratch/project_462000189/hans/ffmpeg/lib/pkgconfig

#pip install <whatever>

export OMP_NUM_THREADS=16
export MPICH_GPU_SUPPORT_ENABLED=1
export RDZV_HOST=$(hostname)
export RDZV_PORT=29400
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK


srun python3 -m torch.distributed.launch \
 --nproc_per_node=1 \
 --nnodes=$SLURM_JOB_NUM_NODES \
 --rdzv_id=$SLURM_JOB_ID \
 --rdzv_backend=c10d \
 --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
 train_net.py \
 --config-file "experiments/HC-STVG/e2e_STCAT_R101_HCSTVG.yaml" \
 --use-seed \


## compare 37000/50000 case with other files
