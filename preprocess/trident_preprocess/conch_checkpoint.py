from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


CONCH_MODEL_NAME = "conch_ViT-B-16"
EMBEDDING_CONTRACT_VERSION = 1
EMBEDDING_CONTRACT_KIND_ATTR = "embedding_contract_kind"
SMOKE_FIXTURE_CONTRACT_KIND = "sample_smoke_fixture"
FIXTURE_ONLY_ATTR = "fixture_only"
CHECKPOINT_FILENAME = "pytorch_model.bin"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ResolvedCheckpoint:
    source: str
    path: Path
    sha256: str


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_hf_reference(reference: str) -> tuple[str, str | None]:
    value = reference.removeprefix("hf_hub:").strip()
    if not value:
        raise ValueError("Hugging Face checkpoint reference is empty")
    if "@" not in value:
        return value, None
    repo_id, revision = value.rsplit("@", 1)
    if not repo_id or not revision:
        raise ValueError(f"Invalid Hugging Face checkpoint reference: {reference!r}")
    return repo_id, revision


def resolve_conch_checkpoint(checkpoint: str | Path) -> ResolvedCheckpoint:
    source = str(checkpoint)
    if source.startswith("hf_hub:"):
        repo_id, revision = parse_hf_reference(source)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise ImportError(
                "huggingface_hub is required to resolve hf_hub checkpoint references"
            ) from error
        resolved = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=CHECKPOINT_FILENAME,
                revision=revision,
            )
        ).resolve()
    else:
        resolved = Path(source).expanduser().resolve()
        if resolved.is_dir():
            resolved = resolved / CHECKPOINT_FILENAME
    if not resolved.is_file():
        raise FileNotFoundError(f"CONCH checkpoint file does not exist: {resolved}")
    return ResolvedCheckpoint(source=source, path=resolved, sha256=sha256_file(resolved))


def valid_checkpoint_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256_PATTERN.fullmatch(value))


def embedding_contract(checkpoint_sha256: str) -> Dict[str, object]:
    if not valid_checkpoint_sha256(checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be 64 lowercase hexadecimal characters")
    return {
        "embedding_contract_version": EMBEDDING_CONTRACT_VERSION,
        "model_name": CONCH_MODEL_NAME,
        "checkpoint_sha256": checkpoint_sha256,
    }


def smoke_fixture_embedding_contract(bundle_sha256: str) -> Dict[str, object]:
    """Return a conspicuously non-production contract for local smoke fixtures."""
    return {
        **embedding_contract(bundle_sha256),
        EMBEDDING_CONTRACT_KIND_ATTR: SMOKE_FIXTURE_CONTRACT_KIND,
        FIXTURE_ONLY_ATTR: True,
    }
