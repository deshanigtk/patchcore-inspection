#!/bin/bash

# The project ID which this job should run under:
#SBATCH --account="punim0512"
# The name of the job:
#SBATCH --job-name="patchcore-mnd"
# Partition for the job:
#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn
# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=05-00:00:00
# Send yourself an email when the job:
#SBATCH --mail-user=dpoddenige@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

# The modules to load:
module load anaconda3/2021.11
eval "$(conda shell.bash hook)"
conda activate ../envs/patchcore-env
module load fosscuda/2020b
module load cudnn/8.9.2.26-cuda-12.0.1


env PYTHONPATH=/data/gpfs/projects/punim0512/deshani_projects/patchcore-inspection/src python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model \
    --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_MND \
    --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 \
    --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 \
    --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 \
    --subdatasets mnd-1024-sample mvtec /data/gpfs/projects/punim0512/deshani_projects/datasets/
