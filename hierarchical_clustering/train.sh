#!/bin/bash
#SBATCH --time 72:00:00 
#SBATCH --mem=100G
#SBATCH -J c_new
#SBATCH -p __REPATH_PRIVATE_SLURM_PARTITION_011__
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -A __REPATH_PRIVATE_SLURM_ACCOUNT_012__
#SBATCH -o clustering.out
#SBATCH --gres=gpu:1

ulimit -n 65535
source __REPATH_PRIVATE_HOME_001__/py310_env/bin/activate
sh run_samples.sh
