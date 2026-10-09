#!/bin/bash
#SBATCH --time 71:59:00
#SBATCH --mem=100G
#SBATCH -J test
#SBATCH -p __REPATH_PRIVATE_SLURM_PARTITION_011__
#SBATCH -c 32
#SBATCH -N 1
#SBATCH -A __REPATH_PRIVATE_SLURM_ACCOUNT_012__
#SBATCH -o step1_5.out
#SBATCH --gres=gpu:1

ulimit -n 65535
source __REPATH_PRIVATE_HOME_001__/py310_env/bin/activate
python run_batch_of_slides.py --task seg --wsi_dir __REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA/svs_files --job_dir __REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA_processed --segmenter hest --batch_size 32
