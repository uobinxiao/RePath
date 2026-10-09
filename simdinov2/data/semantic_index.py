# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict

import numpy as np


SEMANTIC_INDEX_SCHEMA_VERSION = 1
REQUIRED_SAMPLING_BUCKETS = (
    "random_eligible_clusters",
    "bridge_clusters",
    "architecture_bridge_clusters",
    "rare_clean_clusters",
    "tissue_specific_clusters",
)
SAMPLING_BUCKETS = REQUIRED_SAMPLING_BUCKETS + ("architecture_specific_clusters",)
ARRAY_FILES = {
    "fine-records.npy": "fine_records",
    "fine-slide-offsets.npy": "fine_slide_offsets",
    "slide-entry-offsets.npy": "slide_entry_offsets",
    "bucket-masks.npy": "bucket_masks",
}


class SemanticSamplingIndex:
    """Memory-mapped hierarchy produced by ``prepare_offline_sampling.py``."""

    def __init__(self, extra_root: str | Path, split: str) -> None:
        self.root = Path(extra_root).expanduser().resolve()
        self.split = str(split).upper()
        self.index_dir = self.root / f"semantic-index-{self.split}"
        self.manifest_path = self.index_dir / "manifest.json"
        self.manifest = self._load_manifest()

        self.fine_records = self._load_array("fine-records.npy")
        self.fine_slide_offsets = self._load_array("fine-slide-offsets.npy")
        self.slide_entry_offsets = self._load_array("slide-entry-offsets.npy")
        self.bucket_masks = self._load_array("bucket-masks.npy")

        self.bucket_bits = self._validate_bucket_bits(self.manifest.get("bucket_bits"))
        self.compiled_index_id = str(self.manifest.get("compiled_index_id", "")).strip()
        if not re.fullmatch(r"[0-9a-f]{64}", self.compiled_index_id):
            raise ValueError(f"{self.manifest_path}: compiled_index_id must be a SHA-256 digest")
        identity_payload = {
            key: value for key, value in self.manifest.items() if key != "compiled_index_id"
        }
        identity_bytes = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if hashlib.sha256(identity_bytes).hexdigest() != self.compiled_index_id:
            raise ValueError(f"{self.manifest_path}: compiled_index_id does not match the manifest")
        for field in ("config_hash", "input_fingerprint"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(self.manifest["source"].get(field, ""))):
                raise ValueError(f"{self.manifest_path}: source.{field} must be a SHA-256 digest")
        self._validate_arrays()

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Semantic sampling manifest not found: {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError(f"{self.manifest_path}: expected a JSON object")
        if int(manifest.get("schema_version", -1)) != SEMANTIC_INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"{self.manifest_path}: unsupported schema_version={manifest.get('schema_version')}; "
                f"expected {SEMANTIC_INDEX_SCHEMA_VERSION}"
            )
        if str(manifest.get("split", "")).upper() != self.split:
            raise ValueError(f"{self.manifest_path}: split does not match {self.split}")
        if manifest.get("status") != "complete":
            raise ValueError(f"{self.manifest_path}: compiled semantic index is not complete")
        source = manifest.get("source")
        if not isinstance(source, dict) or source.get("profile") != "production" or source.get("approved") is not True:
            raise ValueError(f"{self.manifest_path}: source is not an approved production index")
        return manifest

    def _load_array(self, name: str) -> np.ndarray:
        path = self.index_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Semantic sampling array not found: {path}")
        arrays = self.manifest.get("arrays")
        descriptor = arrays.get(ARRAY_FILES[name]) if isinstance(arrays, dict) else None
        if not isinstance(descriptor, dict) or descriptor.get("path") != name:
            raise ValueError(f"{self.manifest_path}: missing descriptor for {name}")
        expected_digest = str(descriptor.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ValueError(f"{self.manifest_path}: invalid SHA-256 descriptor for {name}")
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise ValueError(f"{path}: SHA-256 does not match the compiled manifest")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != [int(item) for item in descriptor.get("shape", [])]:
            raise ValueError(f"{path}: shape does not match the compiled manifest")
        if str(value.dtype) != str(descriptor.get("dtype")):
            raise ValueError(f"{path}: dtype does not match the compiled manifest")
        return value

    def _validate_bucket_bits(self, value: Any) -> Dict[str, int]:
        if not isinstance(value, dict):
            raise ValueError(f"{self.manifest_path}: bucket_bits must be an object")
        result: Dict[str, int] = {}
        for name in REQUIRED_SAMPLING_BUCKETS:
            if name not in value:
                raise ValueError(f"{self.manifest_path}: bucket_bits is missing {name}")
        for name in SAMPLING_BUCKETS:
            if name not in value:
                continue
            bit = int(value[name])
            if bit < 0 or bit > 7:
                raise ValueError(f"{self.manifest_path}: invalid bit {bit} for {name}")
            result[name] = bit
        if len(set(result.values())) != len(result):
            raise ValueError(f"{self.manifest_path}: bucket bits must be unique")
        return result

    @staticmethod
    def _validate_offsets(name: str, offsets: np.ndarray) -> None:
        if offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if len(offsets) == 0 or int(offsets[0]) != 0:
            raise ValueError(f"{name} must start at zero")
        if np.any(offsets[1:] <= offsets[:-1]):
            raise ValueError(f"{name} must be strictly increasing")

    def _validate_arrays(self) -> None:
        if self.fine_records.ndim != 2 or self.fine_records.shape[1] != 3:
            raise ValueError("fine-records.npy must have shape [F, 3]")
        if not np.issubdtype(self.fine_records.dtype, np.integer):
            raise ValueError("fine-records.npy must use an integer dtype")
        fine_count = int(self.fine_records.shape[0])
        if fine_count == 0:
            raise ValueError("semantic sampling index contains no eligible fine clusters")
        if self.bucket_masks.shape != (fine_count,) or self.bucket_masks.dtype != np.uint8:
            raise ValueError("bucket-masks.npy must be uint8 with one value per fine cluster")
        if self.fine_slide_offsets.shape != (fine_count + 1,):
            raise ValueError("fine-slide-offsets.npy must contain F + 1 values")

        self._validate_offsets("fine-slide-offsets.npy", self.fine_slide_offsets)
        self._validate_offsets("slide-entry-offsets.npy", self.slide_entry_offsets)
        slide_group_count = int(self.fine_slide_offsets[-1])
        if self.slide_entry_offsets.shape != (slide_group_count + 1,):
            raise ValueError("slide-entry-offsets.npy length does not match fine-slide-offsets.npy")

        entry_count = int(self.slide_entry_offsets[-1])
        manifest_entry_count = int(self.manifest.get("entry_count", -1))
        if entry_count <= 0 or manifest_entry_count != entry_count:
            raise ValueError("semantic sidecar entry count does not match its manifest")

        records = np.asarray(self.fine_records, dtype=np.int64)
        order = np.lexsort((records[:, 2], records[:, 1], records[:, 0]))
        if not np.array_equal(order, np.arange(fine_count)):
            raise ValueError("fine records must be sorted by magnification, coarse ID, and fine ID")
        if fine_count > 1 and np.any(np.all(records[1:] == records[:-1], axis=1)):
            raise ValueError("fine records must be unique")

        random_bit = np.uint8(1 << self.bucket_bits["random_eligible_clusters"])
        if np.any((self.bucket_masks & random_bit) == 0):
            raise ValueError("every compiled fine cluster must be random-eligible")

    @property
    def entry_count(self) -> int:
        return int(self.slide_entry_offsets[-1])

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    return digest.hexdigest()
                digest.update(chunk)

    def verify_entry_files(self) -> None:
        descriptor = self.manifest.get("entries")
        if not isinstance(descriptor, dict):
            raise ValueError(f"{self.manifest_path}: missing entries descriptor")
        required = (
            (descriptor, f"entries-{self.split}.npy"),
            (descriptor.get("mapping"), f"entries-{self.split}.json"),
            (descriptor.get("schema"), f"entries-{self.split}.schema.json"),
        )
        for value, expected_name in required:
            if not isinstance(value, dict) or value.get("path") != expected_name:
                raise ValueError(f"{self.manifest_path}: invalid descriptor for {expected_name}")
            expected_digest = str(value.get("sha256", ""))
            path = self.root / expected_name
            if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
                raise ValueError(f"{self.manifest_path}: invalid file contract for {expected_name}")
            if self._sha256_file(path) != expected_digest:
                raise ValueError(f"{path}: SHA-256 does not match the compiled manifest")

    def contract(self) -> Dict[str, Any]:
        source = self.manifest["source"]
        return {
            "compiled_index_id": self.compiled_index_id,
            "entry_count": self.entry_count,
            "source_config_hash": source.get("config_hash"),
            "source_input_fingerprint": source.get("input_fingerprint"),
        }
