#!/bin/bash
#SBATCH --time 23:59:00
#SBATCH --mem=50G
#SBATCH -J mag_40
#SBATCH -p __REPATH_PRIVATE_SLURM_PARTITION_011__
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -A __REPATH_PRIVATE_SLURM_ACCOUNT_012__
#SBATCH -o 3b_mag40_11.out
#SBATCH --gres=gpu:1

ulimit -n 65535
source __REPATH_PRIVATE_HOME_001__/py310_env/bin/activate
python run_batch_of_slides.py --task feat --wsi_dir __REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA/svs_files --job_dir __REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA_processed --patch_encoder conch_v1 --mag 40 --patch_size 256 --max_workers 16
