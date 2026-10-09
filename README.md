# RePath: Towards Efficient and Reproducible Foundation Models for Computational Pathology (Under Review)


## Configure local paths

The released WSI scripts and configurations use placeholders for local paths, slide examples, and SLURM settings. Configure the values needed by your workflow before running preprocessing or training. 

1. Open [`configure_paths.py`](configure_paths.py) and fill in the values in the `REPLACEMENTS` dictionary at the top of the file. 
2. Preview the replacements from the repository root:

   ```shell
   python3 configure_paths.py --dry-run
   ```

3. Apply them to your local source and configuration files:

   ```shell
   python3 configure_paths.py
   ```

Empty values leave their placeholders unchanged. The script reports the files updated and the remaining placeholders; configure those required by the tasks you intend to run. 

## Data preparation

First, you need to download the [TCGA](https://portal.gdc.cancer.gov/analysis_page?app=Downloads) dataset. Set Experimental Strategy as Diagnostic Slide,  Access as Open to filter the slides. 

### Preprocessing with TRIDENT

Install [TRIDENT](https://github.com/mahmoodlab/TRIDENT) in your preprocessing environment and obtain a local CONCH v1 checkpoint. The preprocessing workflow requires a CUDA GPU and OpenSlide, including its native library, for reading SVS files. Install the clustering dependencies in the same environment:

```shell
python3 -m pip install -r hierarchical_clustering/requirements.txt
```

Follow the five preprocessing steps in
[`preprocess/trident_preprocess`](preprocess/trident_preprocess):

1. [Tissue segmentation](preprocess/trident_preprocess/step1_tissue_seg.sh).
2. [Patch coordinate generation](preprocess/trident_preprocess/step2_tissue_patching.sh).
3. [CONCH patch feature extraction](preprocess/trident_preprocess/step3_patch_feature_extraction.sh).
4. [Text-aligned feature projection](preprocess/trident_preprocess/step4_convert_conch_trident_features.py).
5. [HSV tissue filtering](preprocess/trident_preprocess/step5_gen_hsv_filtered_patches.py).

Run **Step 1 once**, then run **Steps 2–5 separately for each magnification: 5x, 10x, 20x, and 40x**. Before processing each magnification, update `--mag` in Steps 2, 3, and 5, and the corresponding input/output paths in Steps 4 and 5. Keep the magnification consistent across all four steps. The Step 2 and Step 3 scripts use 40x as an example; change it for each run.

### Hierarchical clustering and offline semantic index

Prepare `metadata_with_organ.json` as a JSON list with one record per slide. Export the GDC slide metadata and join the corresponding case and patient information. Slide filenames must match the TRIDENT feature files:

```json
[
  {
    "slide_submitter_id": "example_slide",
    "patient_id": "example_patient",
    "file_id": "example_file_id",
    "case_id": "example_case_id",
    "file_name": "example_slide.svs",
    "primary_site": "Bronchus and lung",
    "project_id": "TCGA-LUAD",
    "disease_type": "Lung Adenocarcinoma"
  }
]
```

Configure the paths at the top of [`run_samples.sh`](hierarchical_clustering/run_samples.sh): raw features, projected features, and HSV filters for all four magnifications, plus `WSI_ROOT`, `CHECKPOINT_PATH`, `METADATA_PATH`, and the output paths. Use absolute paths and the same CONCH v1 checkpoint used during preprocessing. Run the script once from the repository root:

```shell
bash hierarchical_clustering/run_samples.sh
```

The script processes 5x, 10x, 20x, and 40x in one run, builds a separate hierarchy for each magnification, encodes the pathology prompts, and runs all eight offline-index stages. It writes the configuration to `CONFIG_OUTPUT` and the index to `OFFLINE_OUTPUT`, as configured in the script.

The first run leaves the index unapproved. Inspect `audit_report.html` and the montages in `OFFLINE_OUTPUT`, then copy `audit_review_template.csv` and fill in `reviewer`, `decision`, and optional `notes`. Keep the cluster identifiers and run-binding columns unchanged. Every selected cluster needs a reviewer and a passing decision (`pass`, `approved`, or `approve`).

Finalize the reviewed index using the `CONFIG_OUTPUT` path and your completed review CSV:

```shell
python3 hierarchical_clustering/run_remote_offline.py finalize \
    --config "/path/to/offline_tcga.json" \
    --review-csv "/path/to/completed_audit_review.csv"
```

Successful finalization records `approved: true` in `OFFLINE_OUTPUT/run_manifest.json`. Use `OFFLINE_OUTPUT` as `--offline-index` in the training preparation command below.

## Training

1. Compile the approved offline index into memory-mapped WSIPatch files once before training. Replace the paths below with your index and compiler output directories:

   ```shell
   python preprocess/extra_file_creation/prepare_offline_sampling.py \
       --offline-index "/path/to/approved/offline_index" \
       --output-dir "/path/to/compiled/wsipatch_extra" \
       --split TRAIN
   ```

2. Prepare initialization weights using [`preprocess/init_weight_transform`](preprocess/init_weight_transform). For SimDINOv2 checkpoints, choose the [ViT-B](preprocess/init_weight_transform/simdinov2_base_transform.py) or [ViT-L](preprocess/init_weight_transform/simdinov2_large_transform.py) script to match your training architecture. For DINOv2 backbone weights, use [`torch_to_fvcore.py`](preprocess/init_weight_transform/torch_to_fvcore.py). Set the input and output checkpoint paths inside the selected script before running it. For example, run the SimDINOv2 ViT-B conversion from the repository root:

   ```shell
   python preprocess/init_weight_transform/simdinov2_base_transform.py
   ```

3. Configure the repository-root [`train.sh`](train.sh): set the `SLURM_*` resources, `--config-file`, and `--output-dir`. In `train.dataset_path`, set `root` to your WSI directory and `extra` to the compiler output directory from Step 1. Keep the WSI paths consistent with the offline index. In the selected YAML, set `MODEL.INIT_WEIGHTS` to the converted checkpoint from Step 2 and adjust training settings such as batch size and `train.OFFICIAL_EPOCH_LENGTH`.

4. Launch training from the repository root:

   ```shell
   bash train.sh
   ```

The script submits the training job to SLURM through Submitit and enables the `semantic_hierarchical` sampler.

## Acknowledgements

This project is largely built upon the orignal [DINO](https://github.com/facebookresearch/dino), [DINOv2](https://github.com/facebookresearch/dinov2) and [SimDINO](https://github.com/RobinWu218/SimDINO) projects.
