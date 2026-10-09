from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .common import (
    allow_smoke_fixture_contract,
    atomic_json,
    begin_stage,
    clustering_path,
    complete_stage,
    fail_stage,
    h5_is_upsampled,
    load_config,
    load_h5_index,
    load_prompt_h5,
    load_upsampled_exclusions,
    metadata_by_slide,
    output_root,
    projected_h5_map,
    read_text_aligned_checkpoint_sha256,
    read_wsi_manifest,
    resolve_wsi_path,
    selected_source_rows,
    slide_submitter_id,
    validate_text_aligned_features,
)


STAGE = "01_build_tile_index"


TILE_SCHEMA = pa.schema(
    [
        ("tile_id", pa.string()),
        ("wsi_id", pa.string()),
        ("slide_id", pa.string()),
        ("patient_id", pa.string()),
        ("file_id", pa.string()),
        ("case_id", pa.string()),
        ("tcga_project", pa.string()),
        ("primary_site", pa.string()),
        ("tissue_or_organ", pa.string()),
        ("disease_type", pa.string()),
        ("magnification", pa.int16()),
        ("scale_bin", pa.string()),
        ("x", pa.int64()),
        ("y", pa.int64()),
        ("mpp", pa.float32()),
        ("patch_size", pa.int32()),
        ("patch_size_level0", pa.int32()),
        ("level0_magnification", pa.int16()),
        ("raw_embedding_h5_path", pa.string()),
        ("projected_embedding_h5_path", pa.string()),
        ("embedding_row_index", pa.int64()),
        ("packed_row_index", pa.int64()),
        ("fine_cluster_id", pa.int32()),
        ("coarse_cluster_id", pa.int32()),
        ("tile_path", pa.string()),
        ("wsi_path", pa.string()),
        ("quality_score", pa.float32()),
        ("tissue_ratio", pa.float32()),
        ("blur_score", pa.float32()),
        ("quality_source", pa.string()),
    ]
)


class RowGroupBufferedWriter:
    """Buffer Arrow tables so each Parquet write honors the configured row-group size."""

    def __init__(self, writer: pq.ParquetWriter, row_group_size: int) -> None:
        if row_group_size <= 0:
            raise ValueError("row_group_size must be positive")
        self.writer = writer
        self.row_group_size = row_group_size
        self._tables: list[pa.Table] = []
        self._rows = 0

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return

        offset = 0
        if self._rows:
            prefix_rows = min(self.row_group_size - self._rows, table.num_rows)
            self._tables.append(table.slice(0, prefix_rows))
            self._rows += prefix_rows
            offset += prefix_rows
        if self._rows == self.row_group_size:
            combined = pa.concat_tables(self._tables)
            self.writer.write_table(combined, row_group_size=self.row_group_size)
            self._tables = []
            self._rows = 0

        remaining_rows = table.num_rows - offset
        full_rows = remaining_rows - (remaining_rows % self.row_group_size)
        if full_rows:
            self.writer.write_table(
                table.slice(offset, full_rows),
                row_group_size=self.row_group_size,
            )
            offset += full_rows

        tail_rows = table.num_rows - offset
        if tail_rows:
            tail = table.slice(offset, tail_rows)
            if table.num_rows > self.row_group_size:
                # Arrow slices retain the complete source buffers. Materialize only
                # this small tail so it cannot pin an arbitrarily large input chunk.
                tail = tail.take(pa.array(np.arange(tail_rows, dtype=np.int64)))
            self._tables = [tail]
            self._rows = tail_rows

    def flush(self) -> None:
        if not self._tables:
            return
        combined = pa.concat_tables(self._tables)
        self.writer.write_table(combined, row_group_size=self.row_group_size)
        self._tables = []
        self._rows = 0


def nullable_float32(size: int) -> pa.Array:
    return pa.nulls(size, type=pa.float32())


def nullable_strings(value: str | None, size: int) -> pa.Array:
    return pa.array([value] * size, type=pa.string())


def build_chunk(
    *,
    shard: Mapping[str, Any],
    metadata: Mapping[str, Any],
    projected_path: Path,
    local_start: int,
    coords: np.ndarray,
    fine_ids: np.ndarray,
    coarse_ids: np.ndarray,
    mag: int,
    wsi_path: str | None,
    embedding_rows: np.ndarray | None = None,
) -> pa.Table:
    size = int(coords.shape[0])
    local_rows = np.arange(local_start, local_start + size, dtype=np.int64)
    if embedding_rows is None:
        embedding_rows = local_rows
    if embedding_rows.shape != (size,):
        raise ValueError("embedding_rows must have one source row per coordinate")
    packed_rows = local_rows + int(shard["row_start"])
    file_name = str(metadata["file_name"])
    wsi_id = Path(file_name).stem
    slide_id = slide_submitter_id(shard)
    tile_ids = [f"{wsi_id}|{mag}x|{int(row)}" for row in embedding_rows]
    values = [
        pa.array(tile_ids, type=pa.string()),
        pa.array([wsi_id] * size, type=pa.string()),
        pa.array([slide_id] * size, type=pa.string()),
        pa.array([str(metadata["patient_id"])] * size, type=pa.string()),
        pa.array([str(metadata["file_id"])] * size, type=pa.string()),
        pa.array([str(metadata["case_id"])] * size, type=pa.string()),
        pa.array([str(metadata["project_id"])] * size, type=pa.string()),
        pa.array([str(metadata["primary_site"])] * size, type=pa.string()),
        pa.array([str(metadata["primary_site"])] * size, type=pa.string()),
        pa.array([str(metadata["disease_type"])] * size, type=pa.string()),
        pa.array(np.full(size, mag, dtype=np.int16), type=pa.int16()),
        pa.array([f"{mag}x"] * size, type=pa.string()),
        pa.array(coords[:, 0].astype(np.int64, copy=False), type=pa.int64()),
        pa.array(coords[:, 1].astype(np.int64, copy=False), type=pa.int64()),
        nullable_float32(size),
        pa.array(np.full(size, int(shard["patch_size"]), dtype=np.int32), type=pa.int32()),
        pa.array(np.full(size, int(shard["patch_size_level0"]), dtype=np.int32), type=pa.int32()),
        pa.array(
            np.full(size, int(shard["level0_magnification"]), dtype=np.int16),
            type=pa.int16(),
        ),
        pa.array([str(Path(shard["path"]).resolve())] * size, type=pa.string()),
        pa.array([str(projected_path.resolve())] * size, type=pa.string()),
        pa.array(embedding_rows, type=pa.int64()),
        pa.array(packed_rows, type=pa.int64()),
        pa.array(fine_ids.astype(np.int32, copy=False), type=pa.int32()),
        pa.array(coarse_ids.astype(np.int32, copy=False), type=pa.int32()),
        nullable_strings(None, size),
        nullable_strings(wsi_path, size),
        nullable_float32(size),
        nullable_float32(size),
        nullable_float32(size),
        pa.array(["none_semantic_artifact_only"] * size, type=pa.string()),
    ]
    return pa.Table.from_arrays(values, schema=TILE_SCHEMA)


def validate_shard_pair(
    raw_path: Path,
    projected_path: Path,
    expected_source_tiles: int,
    prompt_dim: int,
    prompt_checkpoint_sha256: str,
    allow_smoke_fixture: bool = False,
) -> tuple[h5py.File, h5py.File]:
    raw = h5py.File(raw_path, "r")
    projected = h5py.File(projected_path, "r")
    try:
        for handle, label in ((raw, "raw"), (projected, "projected")):
            if "coords" not in handle or "features" not in handle:
                raise ValueError(f"{handle.filename}: {label} H5 requires coords and features")
            if handle["coords"].shape != (expected_source_tiles, 2):
                raise ValueError(f"{handle.filename}: unexpected coords shape {handle['coords'].shape}")
            if handle["features"].shape[0] != expected_source_tiles:
                raise ValueError(f"{handle.filename}: feature row count does not match clustering index")
        validate_text_aligned_features(
            projected["features"],
            projected_path,
            expected_source_tiles,
            prompt_dim,
            expected_checkpoint_sha256=prompt_checkpoint_sha256,
            allow_smoke_fixture=allow_smoke_fixture,
        )
        return raw, projected
    except BaseException:
        raw.close()
        projected.close()
        raise


def validate_projected_inventory(
    projected_map: Mapping[str, Path],
    expected_slide_ids: set[str],
    excluded_upsampled_slide_ids: set[str],
    mag: int,
) -> list[str]:
    overlap = sorted(expected_slide_ids & excluded_upsampled_slide_ids)
    missing = sorted(expected_slide_ids - set(projected_map))
    extra = set(projected_map) - expected_slide_ids
    permitted_extra = extra & excluded_upsampled_slide_ids
    unexpected_extra = sorted(extra - excluded_upsampled_slide_ids)
    invalid_permitted_extra = sorted(
        slide_id
        for slide_id in permitted_extra
        if not h5_is_upsampled(projected_map[slide_id], mag)
    )
    if overlap or missing or unexpected_extra or invalid_permitted_extra:
        raise ValueError(
            f"{mag}x raw/projected slide set mismatch; missing={missing[:5]}, "
            f"extra={unexpected_extra[:5]}, index_and_excluded={overlap[:5]}, "
            f"excluded_metadata_mismatch={invalid_permitted_extra[:5]}"
        )
    return sorted(permitted_extra)


def run(config: Mapping[str, Any], overwrite: bool = False) -> None:
    root = output_root(config)
    dataset_path = root / "tile_index.parquet"
    summary_path = root / "tile_index_summary.json"
    should_run, _ = begin_stage(config, STAGE, [dataset_path, summary_path], overwrite=overwrite)
    if not should_run:
        return

    temporary = root / ".tile_index.parquet.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    progress = None
    try:
        allow_smoke_fixture = allow_smoke_fixture_contract(config)
        metadata_map = metadata_by_slide(config["metadata_path"])
        _, prompt_records = load_prompt_h5(
            config["prompt_embeddings_path"],
            allow_smoke_fixture=allow_smoke_fixture,
        )
        prompt_checkpoint_sha256 = read_text_aligned_checkpoint_sha256(
            config["prompt_embeddings_path"],
            allow_smoke_fixture=allow_smoke_fixture,
        )
        with h5py.File(config["prompt_embeddings_path"], "r") as prompt_h5:
            prompt_dim = int(prompt_h5["features"].shape[1])
        if not prompt_records:
            raise ValueError("Prompt embedding file contains no concepts")
        wsi_manifest = read_wsi_manifest(config["wsi"].get("manifest"))
        fine_level = int(config["levels"]["fine"])
        coarse_level = int(config["levels"]["coarse"])
        chunk_size = int(config["runtime"]["feature_chunk_size"])
        row_group_size = int(config["runtime"]["parquet_row_group_size"])
        summary: Dict[str, Any] = {"schema_version": 1, "magnifications": {}, "total_tiles": 0}
        shards_by_mag = {
            int(mag): load_h5_index(config, int(mag))
            for mag in config["magnifications"]
        }
        total_shards = sum(len(shards) for shards in shards_by_mag.values())
        total_index_tiles = sum(
            int(shard["n_tiles"])
            for shards in shards_by_mag.values()
            for shard in shards
        )
        completed_shards = 0
        progress = tqdm(
            total=total_index_tiles,
            desc="Build tile index",
            unit="tile",
            unit_scale=True,
            dynamic_ncols=True,
            mininterval=1.0,
        )

        for mag in config["magnifications"]:
            shards = shards_by_mag[int(mag)]
            progress.set_postfix_str(
                f"{mag}x files={completed_shards:,}/{total_shards:,}",
                refresh=False,
            )
            projected_map = projected_h5_map(config, mag)
            expected_stems = {str(shard["slide_id"]) for shard in shards}
            excluded_stems = {
                str(record["slide_id"])
                for record in load_upsampled_exclusions(config, mag)
            }
            unused_upsampled = validate_projected_inventory(
                projected_map,
                expected_stems,
                excluded_stems,
                mag,
            )

            fine_assignments = np.load(
                clustering_path(config, mag, f"level{fine_level}", "assignments.npy"), mmap_mode="r"
            )
            coarse_assignments = np.load(
                clustering_path(config, mag, f"level{coarse_level}", "assignments.npy"), mmap_mode="r"
            )
            total_tiles = sum(int(shard["n_tiles"]) for shard in shards)
            if fine_assignments.shape != (total_tiles,) or coarse_assignments.shape != (total_tiles,):
                raise ValueError(f"{mag}x assignment length does not match h5_index total {total_tiles}")

            mag_dir = temporary / f"mag_partition={mag}x"
            mag_dir.mkdir(parents=True)
            parquet_path = mag_dir / "tiles.parquet"
            writer = pq.ParquetWriter(
                parquet_path,
                TILE_SCHEMA,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
            buffered_writer = RowGroupBufferedWriter(writer, row_group_size)
            rows_written = 0
            try:
                for shard in shards:
                    submitter_id = slide_submitter_id(shard)
                    if submitter_id not in metadata_map:
                        raise KeyError(f"{mag}x {shard['slide_id']}: no metadata for {submitter_id}")
                    metadata = metadata_map[submitter_id]
                    projected_path = projected_map[str(shard["slide_id"])]
                    raw_path = Path(shard["path"])
                    raw_h5, projected_h5 = validate_shard_pair(
                        raw_path,
                        projected_path,
                        int(shard.get("source_n_tiles", shard["n_tiles"])),
                        prompt_dim,
                        prompt_checkpoint_sha256,
                        allow_smoke_fixture=allow_smoke_fixture,
                    )
                    wsi_path = resolve_wsi_path(metadata, config, wsi_manifest)
                    try:
                        global_start = int(shard["row_start"])
                        source_rows = selected_source_rows(shard)
                        for local_start in range(0, int(shard["n_tiles"]), chunk_size):
                            local_end = min(int(shard["n_tiles"]), local_start + chunk_size)
                            embedding_rows = source_rows[local_start:local_end]
                            raw_coords = np.asarray(raw_h5["coords"][embedding_rows], dtype=np.int64)
                            projected_coords = np.asarray(
                                projected_h5["coords"][embedding_rows], dtype=np.int64
                            )
                            if not np.array_equal(raw_coords, projected_coords):
                                row_description = (
                                    f"rows {local_start}:{local_end}"
                                    if np.array_equal(
                                        embedding_rows,
                                        np.arange(local_start, local_end, dtype=np.int64),
                                    )
                                    else f"selected source rows {embedding_rows[:5].tolist()}"
                                )
                                raise ValueError(
                                    f"{mag}x {shard['slide_id']}: raw/projected coords differ "
                                    f"at {row_description}"
                                )
                            global_end = global_start + local_end
                            chunk = build_chunk(
                                shard=shard,
                                metadata=metadata,
                                projected_path=projected_path,
                                local_start=local_start,
                                embedding_rows=embedding_rows,
                                coords=raw_coords,
                                fine_ids=np.asarray(
                                    fine_assignments[global_start + local_start : global_end], dtype=np.int32
                                ),
                                coarse_ids=np.asarray(
                                    coarse_assignments[global_start + local_start : global_end], dtype=np.int32
                                ),
                                mag=mag,
                                wsi_path=wsi_path,
                            )
                            buffered_writer.write(chunk)
                            rows_written += len(chunk)
                            progress.update(len(chunk))
                    finally:
                        raw_h5.close()
                        projected_h5.close()
                    completed_shards += 1
                    progress.set_postfix_str(
                        f"{mag}x files={completed_shards:,}/{total_shards:,}",
                        refresh=False,
                    )
                buffered_writer.flush()
            finally:
                writer.close()
            if rows_written != total_tiles:
                raise RuntimeError(f"{mag}x wrote {rows_written} rows, expected {total_tiles}")
            summary["magnifications"][str(mag)] = {
                "slides": len(shards),
                "tiles": total_tiles,
                "unused_upsampled_projected_h5_files": len(unused_upsampled),
                "fine_clusters": int(np.max(fine_assignments)) + 1,
                "coarse_clusters": int(np.max(coarse_assignments)) + 1,
                "parquet": str((dataset_path / f"mag_partition={mag}x" / "tiles.parquet").resolve()),
            }
            summary["total_tiles"] += total_tiles

        temporary.replace(dataset_path)
        atomic_json(summary_path, summary)
        complete_stage(config, STAGE, {"total_tiles": summary["total_tiles"]})
    except BaseException as error:
        if temporary.exists():
            shutil.rmtree(temporary)
        fail_stage(config, STAGE, error)
        raise
    finally:
        if progress is not None:
            progress.close()


def main() -> None:
    from .common import parse_stage_args

    args = parse_stage_args("Build the sampler-ready TCGA tile Parquet index.")
    run(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
