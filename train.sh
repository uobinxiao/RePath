SLURM_NODES=1
SLURM_GPUS_PER_NODE=4
SLURM_TIMEOUT_MIN=20160
SLURM_PARTITION="__REPATH_PRIVATE_SLURM_PARTITION_011__"
SLURM_ACCOUNT="__REPATH_PRIVATE_SLURM_ACCOUNT_012__"
SLURM_JOB_NAME="dinov2_pretraining"
SLURM_MEM_GB=800
SLURM_CPUS_PER_TASK=31

python simdinov2/run/train/train.py \
    --nodes "${SLURM_NODES}" \
    --ngpus "${SLURM_GPUS_PER_NODE}" \
    --timeout "${SLURM_TIMEOUT_MIN}" \
    --partition "${SLURM_PARTITION}" \
    --account "${SLURM_ACCOUNT}" \
    --job-name "${SLURM_JOB_NAME}" \
    --mem-gb "${SLURM_MEM_GB}" \
    --cpus-per-task "${SLURM_CPUS_PER_TASK}" \
    --config-file simdinov2/configs/eval/vitb14_pretrain_bs384_nomag_architecture_specific.yaml \
    --output-dir __REPATH_PRIVATE_PROJECT_ROOT_002__/logs/simdinov2_new_pretrain_new_aug_384_new_clustering_nomag_v3 \
    train.dataset_path=WSIPatch:split=TRAIN:root=__REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA/svs_files/:extra=__REPATH_PRIVATE_PROJECT_ROOT_002__/tcga-extra-clustering_v3 \
    train.sampler.type=semantic_hierarchical
