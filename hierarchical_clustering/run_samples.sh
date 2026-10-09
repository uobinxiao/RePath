#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
HSV_FILTER_WORKERS="${HSV_FILTER_WORKERS:-8}"
HSV_MIN_TISSUE_COVERAGE="${HSV_MIN_TISSUE_COVERAGE:-0.45}"

RAW_5X_DIR="${RAW_5X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/5x_256px_0px_overlap/features_conch_v1}"
RAW_10X_DIR="${RAW_10X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/10x_256px_0px_overlap/features_conch_v1}"
RAW_20X_DIR="${RAW_20X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/20x_256px_0px_overlap/features_conch_v1}"
RAW_40X_DIR="${RAW_40X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/40x_256px_0px_overlap/features_conch_v1}"

HSV_5X_DIR="${HSV_5X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/5x_256px_0px_overlap/5x_hsv_filtered_patches}"
HSV_10X_DIR="${HSV_10X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/10x_256px_0px_overlap/10x_hsv_filtered_patches}"
HSV_20X_DIR="${HSV_20X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/20x_256px_0px_overlap/20x_hsv_filtered_patches}"
HSV_40X_DIR="${HSV_40X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/40x_256px_0px_overlap/40x_hsv_filtered_patches}"

PROJECTED_5X_DIR="${PROJECTED_5X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/5x_256px_0px_overlap/features_conch_v1_text_aligned}"
PROJECTED_10X_DIR="${PROJECTED_10X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/10x_256px_0px_overlap/features_conch_v1_text_aligned}"
PROJECTED_20X_DIR="${PROJECTED_20X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/20x_256px_0px_overlap/features_conch_v1_text_aligned}"
PROJECTED_40X_DIR="${PROJECTED_40X_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_007__/40x_256px_0px_overlap/features_conch_v1_text_aligned}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-__REPATH_PRIVATE_PROJECT_ROOT_002__/path_ad/backbones/weights/conch/pytorch_model.bin}"
METADATA_PATH="${METADATA_PATH:-metadata_with_organ.json}"
WSI_ROOT="${WSI_ROOT:-__REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA/svs_files}"
CLUSTERING_OUTPUT="${CLUSTERING_OUTPUT:-__REPATH_PRIVATE_SCRATCH_ROOT_004__/tcga_conch_v1/clustering}"
PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-__REPATH_PRIVATE_SCRATCH_ROOT_004__/prompt_bank}"
PROMPT_EMBEDDINGS_OUTPUT="${PROMPT_EMBEDDINGS_OUTPUT:-__REPATH_PRIVATE_SCRATCH_ROOT_004__/conch_text_concept_ensemble.h5}"
CONFIG_OUTPUT="${CONFIG_OUTPUT:-__REPATH_PRIVATE_SCRATCH_ROOT_004__/tcga_conch_v1/offline_tcga.json}"
OFFLINE_OUTPUT="${OFFLINE_OUTPUT:-__REPATH_PRIVATE_SCRATCH_ROOT_004__/tcga_conch_v1/offline_index}"

"$PYTHON_BIN" run_remote_offline.py run \
  --raw-feature-dir 5="$RAW_5X_DIR" \
  --raw-feature-dir 10="$RAW_10X_DIR" \
  --raw-feature-dir 20="$RAW_20X_DIR" \
  --raw-feature-dir 40="$RAW_40X_DIR" \
  --hsv-filter-dir 5="$HSV_5X_DIR" \
  --hsv-filter-dir 10="$HSV_10X_DIR" \
  --hsv-filter-dir 20="$HSV_20X_DIR" \
  --hsv-filter-dir 40="$HSV_40X_DIR" \
  --hsv-filter-workers "$HSV_FILTER_WORKERS" \
  --hsv-min-tissue-coverage "$HSV_MIN_TISSUE_COVERAGE" \
  --projected-feature-dir 5="$PROJECTED_5X_DIR" \
  --projected-feature-dir 10="$PROJECTED_10X_DIR" \
  --projected-feature-dir 20="$PROJECTED_20X_DIR" \
  --projected-feature-dir 40="$PROJECTED_40X_DIR" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --metadata-path "$METADATA_PATH" \
  --wsi-root "$WSI_ROOT" \
  --clustering-output "$CLUSTERING_OUTPUT" \
  --prompt-bank-dir "$PROMPT_BANK_DIR" \
  --prompt-embeddings-output "$PROMPT_EMBEDDINGS_OUTPUT" \
  --config-output "$CONFIG_OUTPUT" \
  --offline-output "$OFFLINE_OUTPUT" \
  --device "$DEVICE" \
  --adopt-existing \
  --attest-raw-features-use-checkpoint \
  --overwrite-config
