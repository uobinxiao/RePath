#!/usr/bin/env python3
"""Build deterministic, fixture-only semantic inputs for the local smoke run.

The generated vectors are deliberately not CONCH text-aligned embeddings. They
exercise the offline schemas and stages without asserting unknown checkpoint
provenance. Every output carries a marker that the production loader rejects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

try:
    from .conch_checkpoint import (
        EMBEDDING_CONTRACT_KIND_ATTR,
        FIXTURE_ONLY_ATTR,
        SMOKE_FIXTURE_CONTRACT_KIND,
        sha256_file,
        smoke_fixture_embedding_contract,
    )
except ImportError:
    from conch_checkpoint import (
        EMBEDDING_CONTRACT_KIND_ATTR,
        FIXTURE_ONLY_ATTR,
        SMOKE_FIXTURE_CONTRACT_KIND,
        sha256_file,
        smoke_fixture_embedding_contract,
    )


MAGNIFICATIONS = (5, 10, 20, 40)
FIXTURE_SCHEMA_VERSION = 1
FIXTURE_ALGORITHM = "raw_l2_and_hashed_concepts_v1"
FIXTURE_BUNDLE_ATTR = "fixture_bundle_sha256"
FIXTURE_SOURCE_ATTR = "fixture_source_sha256"
ENSEMBLE_FIELDS = ("polarity", "category", "concept_id")


def parse_mag_dirs(values: Sequence[str], repository_root: Path) -> dict[int, Path]:
    if not values:
        return {
            mag: (
                repository_root
                / "samples"
                / f"{mag}x_256px_0px_overlap"
                / "features_conch_v1"
            ).resolve()
            for mag in MAGNIFICATIONS
        }
    result: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected MAG=DIR for --raw-dir, got {value!r}")
        mag_text, directory = value.split("=", 1)
        mag = int(mag_text)
        if mag not in MAGNIFICATIONS:
            raise ValueError(f"Unsupported smoke magnification: {mag}")
        if mag in result:
            raise ValueError(f"Duplicate --raw-dir for {mag}x")
        result[mag] = Path(directory).expanduser().resolve()
    missing = sorted(set(MAGNIFICATIONS) - set(result))
    if missing:
        raise ValueError(f"Missing --raw-dir entries for {missing}")
    return result


def discover_raw_assets(raw_dirs: Mapping[int, Path]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for mag in MAGNIFICATIONS:
        directory = raw_dirs[mag]
        if not directory.is_dir():
            raise FileNotFoundError(f"Smoke raw feature directory does not exist: {directory}")
        paths = sorted(directory.glob("*.h5"))
        if not paths:
            raise FileNotFoundError(f"No H5 smoke features found in {directory}")
        seen_names: set[str] = set()
        for path in paths:
            if path.name in seen_names:
                raise ValueError(f"Duplicate raw smoke filename for {mag}x: {path.name}")
            seen_names.add(path.name)
            assets.append(
                {
                    "key": f"raw/{mag}x/{path.name}",
                    "magnification": mag,
                    "path": path.resolve(),
                    "output": Path("projected") / f"{mag}x" / path.name,
                }
            )
    return assets


def build_source_inventory(
    raw_assets: Sequence[Mapping[str, Any]],
    concept_bank_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    if not concept_bank_path.is_file():
        raise FileNotFoundError(f"Smoke concept bank does not exist: {concept_bank_path}")
    records: list[dict[str, Any]] = []
    sources = [(str(asset["key"]), Path(asset["path"])) for asset in raw_assets]
    sources.append(("concept_bank", concept_bank_path.resolve()))
    for key, path in sorted(sources):
        records.append(
            {
                "key": key,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    digest.update(FIXTURE_ALGORITHM.encode("utf-8"))
    digest.update(b"\0")
    for record in records:
        digest.update(str(record["key"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\0")
    return records, digest.hexdigest()


def apply_fixture_contract(
    root_attrs: h5py.AttributeManager,
    feature_attrs: h5py.AttributeManager,
    bundle_sha256: str,
    source_sha256: str,
) -> None:
    contract = smoke_fixture_embedding_contract(bundle_sha256)
    for attributes in (root_attrs, feature_attrs):
        for key, value in contract.items():
            attributes[key] = value
        attributes[FIXTURE_BUNDLE_ATTR] = bundle_sha256
        attributes[FIXTURE_SOURCE_ATTR] = source_sha256


def write_projected_fixture(
    source_path: Path,
    destination_path: Path,
    bundle_sha256: str,
    source_sha256: str,
    batch_size: int,
) -> int:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source:
        if "features" not in source or "coords" not in source:
            raise ValueError(f"{source_path}: raw smoke H5 requires features and coords")
        raw_features = source["features"]
        if raw_features.ndim != 2:
            raise ValueError(f"{source_path}: raw smoke features must be 2-D")
        n_rows, feature_dim = (int(value) for value in raw_features.shape)
        if source["coords"].shape != (n_rows, 2):
            raise ValueError(f"{source_path}: coords do not align with raw features")
        with h5py.File(destination_path, "w") as destination:
            for key, value in source.attrs.items():
                destination.attrs[key] = value
            for key in source.keys():
                if key != "features":
                    source.copy(key, destination)
            options: dict[str, Any] = {
                "shape": (n_rows, feature_dim),
                "dtype": "float32",
            }
            if n_rows:
                options.update(
                    {
                        "chunks": (min(batch_size, n_rows), feature_dim),
                        "compression": "gzip",
                    }
                )
            output = destination.create_dataset("features", **options)
            for key, value in raw_features.attrs.items():
                output.attrs[key] = value
            output.attrs["embedding_space"] = "conch_v1_text_aligned"
            output.attrs["normalized"] = True
            output.attrs["source_features"] = "sample_smoke_raw_l2_fixture"
            apply_fixture_contract(
                destination.attrs,
                output.attrs,
                bundle_sha256,
                source_sha256,
            )
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                values = np.asarray(raw_features[start:end], dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"{source_path}: raw smoke features contain non-finite values")
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                if np.any(norms <= 0):
                    raise ValueError(f"{source_path}: raw smoke features contain zero vectors")
                output[start:end] = values / norms
    return feature_dim


def deterministic_unit_vector(key: str, dimension: int) -> np.ndarray:
    payload = bytearray()
    counter = 0
    while len(payload) < dimension * 4:
        digest = hashlib.sha256(f"{key}\0{counter}".encode("utf-8")).digest()
        payload.extend(digest)
        counter += 1
    integers = np.frombuffer(bytes(payload[: dimension * 4]), dtype="<u4")
    values = (integers.astype(np.float64) + 0.5) / float(2**32)
    values = (values * 2.0 - 1.0).astype(np.float32)
    return values / np.linalg.norm(values)


def concept_records(concept_bank_path: Path) -> list[dict[str, str]]:
    with open(concept_bank_path, "r", encoding="utf-8") as handle:
        bank = json.load(handle)
    concepts = bank.get("concepts") if isinstance(bank, dict) else None
    if not isinstance(concepts, dict):
        raise ValueError(f"{concept_bank_path}: expected a concepts mapping")
    records: list[dict[str, str]] = []
    for polarity, categories in concepts.items():
        if not isinstance(categories, dict):
            raise ValueError(f"{concept_bank_path}: polarity {polarity!r} must map categories")
        for category, values in categories.items():
            if not isinstance(values, list):
                raise ValueError(f"{concept_bank_path}: category {category!r} must be a list")
            for index, concept in enumerate(values):
                concept_text = str(concept).strip()
                if not concept_text:
                    raise ValueError(f"{concept_bank_path}: empty concept in {category!r}")
                concept_id = f"{category}_{index:03d}"
                records.append(
                    {
                        "prompt_id": f"{polarity}.{concept_id}",
                        "prompt": concept_text,
                        "concept": concept_text,
                        "concept_id": concept_id,
                        "polarity": str(polarity),
                        "category": str(category),
                    }
                )
    if not records:
        raise ValueError(f"{concept_bank_path}: concept bank is empty")
    return records


def write_prompt_fixture(
    destination_path: Path,
    concept_bank_path: Path,
    dimension: int,
    bundle_sha256: str,
    source_sha256: str,
) -> int:
    records = concept_records(concept_bank_path)
    vectors = np.stack(
        [
            deterministic_unit_vector(
                "\0".join(record[field] for field in ("polarity", "category", "concept_id")),
                dimension,
            )
            for record in records
        ]
    ).astype(np.float32, copy=False)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(destination_path, "w") as destination:
        features = destination.create_dataset("features", data=vectors)
        features.attrs["embedding_space"] = "conch_v1_text_aligned"
        features.attrs["normalized"] = True
        apply_fixture_contract(
            destination.attrs,
            features.attrs,
            bundle_sha256,
            source_sha256,
        )
        destination.create_dataset(
            "prompt_ids",
            data=np.asarray([record["prompt_id"] for record in records], dtype=object),
            dtype=string_dtype,
        )
        destination.create_dataset(
            "prompts",
            data=np.asarray([record["prompt"] for record in records], dtype=object),
            dtype=string_dtype,
        )
        metadata = destination.create_group("metadata")
        for field in ("concept", "concept_id", "polarity", "category"):
            metadata.create_dataset(
                field,
                data=np.asarray([record[field] for record in records], dtype=object),
                dtype=string_dtype,
            )
        destination.attrs["model"] = "fixture-only-deterministic-vectors"
        destination.attrs["checkpoint_path"] = "fixture-only:no-checkpoint"
        destination.attrs["embedding_type"] = "text"
        destination.attrs["normalized"] = True
        destination.attrs["ensemble"] = True
        destination.attrs["ensemble_by"] = ",".join(ENSEMBLE_FIELDS)
    return len(records)


def h5_has_fixture_contract(path: Path, bundle_sha256: str) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            if "features" not in handle:
                return False
            features = handle["features"]
            for attributes in (handle.attrs, features.attrs):
                if attributes.get(EMBEDDING_CONTRACT_KIND_ATTR) != SMOKE_FIXTURE_CONTRACT_KIND:
                    return False
                if bool(attributes.get(FIXTURE_ONLY_ATTR)) is not True:
                    return False
                if attributes.get(FIXTURE_BUNDLE_ATTR) != bundle_sha256:
                    return False
                if attributes.get("checkpoint_sha256") != bundle_sha256:
                    return False
            return True
    except (OSError, ValueError):
        return False


def reusable_bundle(
    output_dir: Path,
    bundle_sha256: str,
    raw_assets: Sequence[Mapping[str, Any]],
) -> bool:
    manifest_path = output_dir / "fixture_manifest.json"
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False
    if (
        manifest.get("schema_version") != FIXTURE_SCHEMA_VERSION
        or manifest.get("algorithm") != FIXTURE_ALGORITHM
        or manifest.get("fixture_bundle_sha256") != bundle_sha256
        or manifest.get("fixture_only") is not True
    ):
        return False
    expected_paths = [output_dir / Path(asset["output"]) for asset in raw_assets]
    expected_paths.append(output_dir / "prompt_embeddings.h5")
    return all(h5_has_fixture_contract(path, bundle_sha256) for path in expected_paths)


def replace_directory(source: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.old.{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.replace(backup)
    try:
        source.replace(destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def prepare_smoke_inputs(
    raw_dirs: Mapping[int, Path],
    concept_bank_path: Path,
    output_dir: Path,
    batch_size: int = 8192,
    overwrite: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_dir = output_dir.expanduser().resolve()
    concept_bank_path = concept_bank_path.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Smoke fixture output must be a directory: {output_dir}")
    protected_paths = [concept_bank_path, *(path.resolve() for path in raw_dirs.values())]
    for protected in protected_paths:
        if (
            output_dir == protected
            or output_dir in protected.parents
            or protected in output_dir.parents
        ):
            raise ValueError(
                f"Smoke fixture output {output_dir} must not overlap source asset {protected}"
            )
    raw_assets = discover_raw_assets(raw_dirs)
    inventory, bundle_sha256 = build_source_inventory(raw_assets, concept_bank_path)
    if not overwrite and reusable_bundle(output_dir, bundle_sha256, raw_assets):
        with open(output_dir / "fixture_manifest.json", "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        print(f"Reusing current smoke fixture bundle: {output_dir}")
        return manifest

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build.", dir=output_dir.parent)
    )
    source_by_key = {record["key"]: record for record in inventory}
    try:
        feature_dimension: int | None = None
        output_records: list[dict[str, Any]] = []
        for asset in raw_assets:
            destination = temporary / Path(asset["output"])
            source_record = source_by_key[str(asset["key"])]
            dimension = write_projected_fixture(
                Path(asset["path"]),
                destination,
                bundle_sha256,
                str(source_record["sha256"]),
                batch_size,
            )
            if feature_dimension is None:
                feature_dimension = dimension
            elif dimension != feature_dimension:
                raise ValueError(
                    f"Raw smoke feature dimensions differ: {feature_dimension} and {dimension}"
                )
            output_records.append(
                {
                    "source_key": str(asset["key"]),
                    "path": str(Path(asset["output"])),
                }
            )
        if feature_dimension is None or feature_dimension <= 0:
            raise ValueError("Smoke raw features have no usable embedding dimension")
        concept_source = source_by_key["concept_bank"]
        prompt_count = write_prompt_fixture(
            temporary / "prompt_embeddings.h5",
            concept_bank_path,
            feature_dimension,
            bundle_sha256,
            str(concept_source["sha256"]),
        )
        output_records.append({"source_key": "concept_bank", "path": "prompt_embeddings.h5"})
        manifest = {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "fixture_only": True,
            "algorithm": FIXTURE_ALGORITHM,
            "fixture_bundle_sha256": bundle_sha256,
            "feature_dimension": feature_dimension,
            "prompt_count": prompt_count,
            "sources": inventory,
            "outputs": output_records,
        }
        with open(temporary / "fixture_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        replace_directory(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(f"Prepared fixture-only smoke embeddings: {output_dir}")
    return manifest


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        action="append",
        default=[],
        metavar="MAG=DIR",
        help="Raw sample H5 directory; provide all four magnifications or omit for defaults.",
    )
    parser.add_argument(
        "--concept-bank",
        default=str(repository_root / "tools" / "prompt_bank" / "concept_bank.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(repository_root / "tmp" / "offline_smoke_inputs"),
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    raw_dirs = parse_mag_dirs(args.raw_dir, repository_root)
    prepare_smoke_inputs(
        raw_dirs=raw_dirs,
        concept_bank_path=Path(args.concept_bank).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
