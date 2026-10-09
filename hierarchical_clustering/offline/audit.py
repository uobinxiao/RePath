from __future__ import annotations

import html
import json
import math
import shutil
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from tqdm import tqdm

from .buckets import BUCKET_COLUMNS
from .common import (
    AUDIT_REVIEW_BINDING_FIELDS,
    atomic_csv,
    atomic_parquet,
    audit_review_binding,
    begin_stage,
    complete_stage,
    fail_stage,
    load_config,
    load_manifest,
    output_root,
    sha256_file,
)


STAGE = "07_make_cluster_audit_montages"


AUDIT_BUCKETS = {
    "bridge": ("is_bridge", "bridge_score"),
    "architecture_bridge": ("is_architecture_bridge", "architecture_bridge_score"),
    "architecture_specific": (
        "is_architecture_specific",
        "architecture_specific_score",
    ),
    "rare_clean": ("is_rare_clean", "rare_clean_score"),
    "tissue_specific": ("is_tissue_specific", "tissue_specific_score"),
    "artifact": (None, "artifact_score"),
    "random": ("is_random_eligible", None),
}


def select_audit_clusters(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    limit = int(config["audit"]["clusters_per_bucket"])
    seed = int(config["audit"]["seed"])
    selections = []
    for mag, mag_frame in frame.groupby("magnification", sort=True):
        for bucket_name, (flag, score) in AUDIT_BUCKETS.items():
            candidates = mag_frame if flag is None else mag_frame.loc[mag_frame[flag]]
            if score is None:
                selected = candidates.sample(
                    n=min(limit, len(candidates)),
                    random_state=seed + int(mag),
                    replace=False,
                ) if len(candidates) else candidates
            else:
                selected = candidates.sort_values(
                    [score, "fine_cluster_id"], ascending=[False, True]
                ).head(limit)
            selected = selected.copy()
            selected["audit_bucket"] = bucket_name
            selected["audit_rank"] = range(1, len(selected) + 1)
            selections.append(selected)
    if not selections:
        return frame.iloc[0:0].assign(audit_bucket=pd.Series(dtype=str), audit_rank=pd.Series(dtype=int))
    return pd.concat(selections, ignore_index=True)


def concept_names(value: str) -> str:
    try:
        records = json.loads(value)
        return ", ".join(str(record.get("concept", record.get("concept_id", ""))) for record in records)
    except (TypeError, json.JSONDecodeError):
        return str(value)


def load_openslide():
    try:
        import openslide
    except (ImportError, OSError) as error:
        raise ImportError(
            "openslide-python is required to render WSI audit montages. "
            "Install OpenSlide and openslide-python, or set audit.manifest_only=true."
        ) from error
    return openslide


def extract_region(slide, representative: Mapping[str, Any], display_size: int) -> Image.Image:
    source_size = int(representative["patch_size_level0"])
    if source_size <= 0 or display_size <= 0:
        raise ValueError("Audit source and display sizes must be positive")
    desired_downsample = max(1.0, source_size / float(display_size))
    level = int(slide.get_best_level_for_downsample(desired_downsample))
    level_downsamples = tuple(float(value) for value in slide.level_downsamples)
    if not level_downsamples:
        raise ValueError("OpenSlide pyramid has no levels")
    level = min(max(level, 0), len(level_downsamples) - 1)
    level_downsample = level_downsamples[level]
    if not math.isfinite(level_downsample) or level_downsample <= 0:
        raise ValueError(f"Invalid OpenSlide level downsample: {level_downsample}")
    level_size = max(1, math.ceil(source_size / level_downsample))
    region = slide.read_region(
        (int(representative["x"]), int(representative["y"])),
        level,
        (level_size, level_size),
    ).convert("RGB")
    if region.size != (display_size, display_size):
        region = region.resize((display_size, display_size), Image.Resampling.LANCZOS)
    return region


class SlideCache:
    def __init__(self, openslide_module, max_open: int) -> None:
        self.openslide_module = openslide_module
        self.max_open = int(max_open)
        self.slides: OrderedDict[str, Any] = OrderedDict()

    def get(self, path: str):
        if path in self.slides:
            slide = self.slides.pop(path)
            self.slides[path] = slide
            return slide
        slide = self.openslide_module.OpenSlide(path)
        self.slides[path] = slide
        while len(self.slides) > self.max_open:
            _, oldest = self.slides.popitem(last=False)
            oldest.close()
        return slide

    def close(self) -> None:
        for slide in self.slides.values():
            try:
                slide.close()
            except Exception:
                pass
        self.slides.clear()


def montage_image(
    cluster: Mapping[str, Any],
    representatives: pd.DataFrame,
    slide_cache: SlideCache,
    display_size: int,
) -> tuple[Image.Image, int]:
    columns = 5
    rows = max(1, math.ceil(max(1, len(representatives)) / columns))
    header_height = 150
    canvas = Image.new("RGB", (columns * display_size, header_height + rows * display_size), "white")
    draw = ImageDraw.Draw(canvas)
    header_lines = [
        f"cluster={cluster['cluster_key']} coarse={int(cluster['coarse_cluster_id'])}",
        (
            f"tiles={int(cluster['n_tiles'])} patients={int(cluster['n_patients'])} "
            f"slides={int(cluster['n_slides'])} entropy={float(cluster['tissue_entropy']):.3f}"
        ),
        (
            f"bridge={float(cluster['bridge_score']):.4f} "
            f"arch_bridge={float(cluster['architecture_bridge_score']):.4f} "
            f"arch_specific={float(cluster['architecture_specific_score']):.4f}"
        ),
        f"rare={float(cluster['rare_clean_score']):.4f} artifact={float(cluster['artifact_score']):.3f}",
        f"positive: {concept_names(cluster['top_positive_concepts'])}",
        f"artifact: {concept_names(cluster['top_artifact_concepts'])}",
    ]
    y = 5
    for line in header_lines:
        draw.text((8, y), line[:210], fill="black")
        y += 26

    failures = 0
    if representatives.empty:
        failures = 1
        draw.text((8, header_height + 8), "no representative tiles", fill="red")
    for tile_index, (_, representative) in enumerate(representatives.iterrows()):
        x_offset = (tile_index % columns) * display_size
        y_offset = header_height + (tile_index // columns) * display_size
        wsi_path = representative.get("wsi_path")
        try:
            if not isinstance(wsi_path, str) or not wsi_path or not Path(wsi_path).is_file():
                raise FileNotFoundError(str(wsi_path))
            tile = extract_region(slide_cache.get(wsi_path), representative, display_size)
            canvas.paste(tile, (x_offset, y_offset))
            draw.rectangle((x_offset, y_offset, x_offset + 105, y_offset + 18), fill="white")
            draw.text(
                (x_offset + 2, y_offset + 2),
                f"r{int(representative['representative_rank'])} {representative['slide_id'][:12]}",
                fill="black",
            )
        except Exception as error:
            failures += 1
            draw.rectangle(
                (x_offset, y_offset, x_offset + display_size - 1, y_offset + display_size - 1),
                outline="red",
                width=3,
            )
            draw.text((x_offset + 8, y_offset + 8), f"missing tile\n{type(error).__name__}", fill="red")
    return canvas, failures


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def render_html(selection: pd.DataFrame) -> str:
    rows = []
    for _, record in selection.sort_values(["audit_bucket", "magnification", "audit_rank"]).iterrows():
        image_path = record.get("montage_path")
        image_cell = (
            f'<a href="{html.escape(str(image_path))}"><img src="{html.escape(str(image_path))}" width="360"></a>'
            if isinstance(image_path, str) and image_path
            else html.escape(str(record.get("render_status", "manifest_only")))
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record['audit_bucket']))}</td>"
            f"<td>{int(record['magnification'])}x</td>"
            f"<td>{html.escape(str(record['cluster_key']))}<br>coarse {int(record['coarse_cluster_id'])}</td>"
            f"<td>{image_cell}</td>"
            f"<td>tiles {int(record['n_tiles'])}<br>patients {int(record['n_patients'])}<br>slides {int(record['n_slides'])}"
            f"<br>coverage {int(record['tissue_coverage'])}<br>entropy {float(record['tissue_entropy']):.3f}</td>"
            f"<td>{html.escape(concept_names(record['top_positive_concepts']))}</td>"
            f"<td>{html.escape(concept_names(record['top_artifact_concepts']))}</td>"
            f"<td><code>{html.escape(str(record['tissue_distribution']))}</code></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Offline cluster audit</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #eee; }}
code {{ white-space: pre-wrap; font-size: 10px; }}
</style></head><body>
<h1>CONCH-guided offline cluster audit</h1>
<p>Review morphology, semantic tags, artifacts, and source-distribution dominance before approving the index.</p>
<table><thead><tr><th>bucket</th><th>scale</th><th>cluster</th><th>montage</th><th>support</th>
<th>positive concepts</th><th>artifact concepts</th><th>tissue distribution</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""


def group_counts(batch: pd.DataFrame, field: str) -> Iterable[tuple[int, str, int]]:
    grouped = batch.groupby(["fine_cluster_id", field], sort=False, observed=True).size()
    for (fine_cluster_id, value), count in grouped.items():
        yield int(fine_cluster_id), str(value), int(count)


def bucket_dominance(
    frame: pd.DataFrame,
    tile_index_dir: Path,
    batch_size: int,
) -> pd.DataFrame:
    records = []
    magnification_groups = list(frame.groupby("magnification", sort=True))
    parquet_files = {
        int(mag): pq.ParquetFile(
            tile_index_dir / f"mag_partition={int(mag)}x" / "tiles.parquet"
        )
        for mag, _ in magnification_groups
    }
    total_tiles = sum(parquet_file.metadata.num_rows for parquet_file in parquet_files.values())
    with tqdm(
        total=total_tiles,
        desc="Audit dominance",
        unit="tile",
        unit_scale=True,
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress:
        for mag, group in magnification_groups:
            mag = int(mag)
            progress.set_postfix_str(f"{mag}x", refresh=False)
            memberships: Dict[int, list[str]] = {}
            for _, row in group.iterrows():
                memberships[int(row["fine_cluster_id"])] = [
                    bucket_name for bucket_name, column in BUCKET_COLUMNS.items() if bool(row[column])
                ]
            counters: Dict[str, Dict[str, Counter[str]]] = {
                bucket_name: {
                    "primary_site": Counter(),
                    "patient_id": Counter(),
                    "slide_id": Counter(),
                    "tcga_project": Counter(),
                }
                for bucket_name in BUCKET_COLUMNS
            }
            parquet_file = parquet_files[mag]
            columns = ["fine_cluster_id", "primary_site", "patient_id", "slide_id", "tcga_project"]
            for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
                batch_frame = batch.to_pandas()
                for field in columns[1:]:
                    for fine_cluster_id, value, count in group_counts(batch_frame, field):
                        for bucket_name in memberships.get(fine_cluster_id, []):
                            counters[bucket_name][field][value] += count
                progress.update(batch.num_rows)

            for bucket_name, column in BUCKET_COLUMNS.items():
                selected = group.loc[group[column]]
                bucket_tile_count = int(selected["n_tiles"].sum())
                record: Dict[str, Any] = {
                    "magnification": mag,
                    "bucket": bucket_name,
                    "n_clusters": int(len(selected)),
                    "n_tiles": bucket_tile_count,
                }
                for field, counter in counters[bucket_name].items():
                    if counter:
                        dominant, count = counter.most_common(1)[0]
                        record[f"dominant_{field}"] = dominant
                        record[f"dominant_{field}_fraction"] = count / max(1, bucket_tile_count)
                    else:
                        record[f"dominant_{field}"] = None
                        record[f"dominant_{field}_fraction"] = 0.0
                records.append(record)
    return pd.DataFrame.from_records(records)


def run(config: Mapping[str, Any], overwrite: bool = False) -> None:
    root = output_root(config)
    montage_dir = root / "audit_montages"
    selection_path = root / "audit_selection.parquet"
    dominance_path = root / "bucket_dominance_report.parquet"
    report_path = root / "audit_report.html"
    review_path = root / "audit_review_template.csv"
    outputs = [montage_dir, selection_path, dominance_path, report_path, review_path]
    should_run, _ = begin_stage(config, STAGE, outputs, overwrite=overwrite)
    if not should_run:
        return

    temporary_montages = root / ".audit_montages.tmp"
    if temporary_montages.exists():
        shutil.rmtree(temporary_montages)
    temporary_montages.mkdir(parents=True)
    slide_cache: SlideCache | None = None
    try:
        buckets = pd.read_parquet(root / "fine_cluster_buckets.parquet")
        representatives = pd.read_parquet(root / "representative_tiles.parquet")
        selection = select_audit_clusters(buckets, config)
        manifest_only = bool(config["audit"]["manifest_only"])
        openslide_module = None if manifest_only else load_openslide()
        if openslide_module is not None:
            slide_cache = SlideCache(openslide_module, int(config["audit"]["max_open_wsi"]))
        render_statuses = []
        montage_paths = []
        montage_sha256s = []
        total_failures = 0
        with tqdm(
            total=len(selection),
            desc="Audit montages",
            unit="cluster",
            dynamic_ncols=True,
            mininterval=1.0,
        ) as progress:
            for _, cluster in selection.iterrows():
                bucket_name = str(cluster["audit_bucket"])
                mag = int(cluster["magnification"])
                cluster_id = int(cluster["fine_cluster_id"])
                progress.set_postfix_str(
                    f"{bucket_name} {mag}x cluster={cluster_id}",
                    refresh=False,
                )
                cluster_representatives = representatives.loc[
                    (representatives["magnification"] == mag)
                    & (representatives["fine_cluster_id"] == cluster_id)
                ].sort_values("representative_rank")
                relative_path = (
                    Path("audit_montages")
                    / bucket_name
                    / f"{mag}x"
                    / f"cluster_{cluster_id}.png"
                )
                temporary_path = (
                    temporary_montages
                    / bucket_name
                    / f"{mag}x"
                    / f"cluster_{cluster_id}.png"
                )
                if manifest_only:
                    render_statuses.append("manifest_only")
                    montage_paths.append(None)
                    montage_sha256s.append(None)
                    progress.update(1)
                    continue
                temporary_path.parent.mkdir(parents=True, exist_ok=True)
                image, failures = montage_image(
                    cluster,
                    cluster_representatives,
                    slide_cache,
                    int(config["audit"]["tile_display_size"]),
                )
                image.save(temporary_path, format="PNG", optimize=True)
                total_failures += failures
                render_statuses.append(
                    "rendered" if failures == 0 else f"rendered_with_{failures}_missing"
                )
                montage_paths.append(str(relative_path))
                montage_sha256s.append(sha256_file(temporary_path))
                progress.update(1)
        selection = selection.copy()
        selection["render_status"] = render_statuses
        selection["montage_path"] = montage_paths
        selection["montage_sha256"] = montage_sha256s
        if total_failures and bool(config["audit"]["strict_wsi"]):
            raise RuntimeError(f"Audit rendering failed for {total_failures} representative tiles")

        dominance = bucket_dominance(
            buckets,
            root / "tile_index.parquet",
            int(config["audit"]["dominance_batch_size"]),
        )
        if montage_dir.exists():
            shutil.rmtree(montage_dir)
        temporary_montages.replace(montage_dir)
        atomic_parquet(selection, selection_path, int(config["runtime"]["parquet_row_group_size"]))
        atomic_parquet(dominance, dominance_path, int(config["runtime"]["parquet_row_group_size"]))
        write_text_atomic(report_path, render_html(selection))
        review_binding = audit_review_binding(load_manifest(config), selection_path)
        review_rows = []
        for cluster_key, group in selection.groupby("cluster_key", sort=True):
            first = group.iloc[0]
            review_rows.append(
                {
                    "cluster_key": cluster_key,
                    "magnification": int(first["magnification"]),
                    "fine_cluster_id": int(first["fine_cluster_id"]),
                    "audit_buckets": ",".join(sorted(group["audit_bucket"].unique())),
                    **review_binding,
                    "reviewer": "",
                    "decision": "",
                    "notes": "",
                }
            )
        review_columns = [
            "cluster_key",
            "magnification",
            "fine_cluster_id",
            "audit_buckets",
            *AUDIT_REVIEW_BINDING_FIELDS,
            "reviewer",
            "decision",
            "notes",
        ]
        atomic_csv(
            review_path,
            review_columns,
            review_rows,
        )
        complete_stage(
            config,
            STAGE,
            {
                "audit_cluster_rows": len(selection),
                "unique_clusters": int(selection["cluster_key"].nunique()),
                "render_failures": total_failures,
                "manifest_only": manifest_only,
            },
        )
    except BaseException as error:
        if temporary_montages.exists():
            shutil.rmtree(temporary_montages)
        fail_stage(config, STAGE, error)
        raise
    finally:
        if slide_cache is not None:
            slide_cache.close()


def main() -> None:
    from .common import parse_stage_args

    args = parse_stage_args("Create WSI audit montages, HTML, and dominance reports.")
    run(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
