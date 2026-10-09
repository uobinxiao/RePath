import argparse
import csv
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import torch

try:
    from .conch_checkpoint import embedding_contract, resolve_conch_checkpoint
except ImportError:
    from conch_checkpoint import embedding_contract, resolve_conch_checkpoint


@dataclass
class PromptRecord:
    prompt_id: str
    prompt: str
    metadata: Dict[str, str] = field(default_factory=dict)


PROMPT_GEN_METADATA_FIELDS = [
    "polarity",
    "category",
    "concept",
    "concept_id",
    "ensemble_size",
    "ensemble_prompt_ids",
    "ensemble_prompts",
    "template_id",
    "template",
]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_conch_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    try:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    except ImportError as exc:
        raise ImportError(
            "CONCH is required. Install it with: "
            "pip install git+https://github.com/Mahmoodlab/CONCH.git"
        ) from exc

    model, _ = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=checkpoint_path,
    )
    model = model.to(device).eval()
    tokenizer = get_tokenizer()
    return model, tokenizer, tokenize


def records_from_json(path: Path) -> List[PromptRecord]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[PromptRecord] = []
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, str):
                records.append(PromptRecord(prompt_id=str(idx), prompt=item))
            elif isinstance(item, dict):
                prompt = item.get("prompt") or item.get("text")
                if not isinstance(prompt, str):
                    raise ValueError(f"{path}: JSON item {idx} must contain a string 'prompt' or 'text'")
                prompt_id = item.get("prompt_id", item.get("id", idx))
                records.append(
                    PromptRecord(
                        prompt_id=str(prompt_id),
                        prompt=prompt,
                        metadata=metadata_from_mapping(item),
                    )
                )
            else:
                raise ValueError(f"{path}: JSON list items must be strings or objects")
    elif isinstance(data, dict):
        for prompt_id, prompt in data.items():
            if not isinstance(prompt, str):
                raise ValueError(f"{path}: JSON object values must be prompt strings")
            records.append(PromptRecord(prompt_id=str(prompt_id), prompt=prompt))
    else:
        raise ValueError(f"{path}: expected a JSON list or object")

    return records


def metadata_from_mapping(row: Dict[str, object]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for key, value in row.items():
        if key in {"prompt", "text"}:
            continue
        if value is None:
            metadata[key] = ""
        else:
            metadata[key] = str(value)
    return metadata


def prompt_record_from_mapping(row: Dict[str, object], fallback_id: str, source: Path) -> PromptRecord:
    prompt = row.get("prompt") or row.get("text")
    if not isinstance(prompt, str):
        raise ValueError(f"{source}: each record must contain a string 'prompt' or 'text'")
    prompt_id = row.get("prompt_id", row.get("id", fallback_id))
    return PromptRecord(
        prompt_id=str(prompt_id),
        prompt=prompt,
        metadata=metadata_from_mapping(row),
    )


def records_from_jsonl(path: Path) -> List[PromptRecord]:
    records: List[PromptRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} must be a JSON object")
            records.append(prompt_record_from_mapping(row, str(line_number), path))
    return records


def records_from_csv(path: Path) -> List[PromptRecord]:
    records: List[PromptRecord] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV file is missing a header row")
        for row_number, row in enumerate(reader, start=2):
            records.append(prompt_record_from_mapping(row, str(row_number), path))
    return records


def records_from_text(path: Path) -> List[PromptRecord]:
    records: List[PromptRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            prompt = line.strip()
            if prompt and not prompt.startswith("#"):
                records.append(PromptRecord(prompt_id=str(line_number), prompt=prompt))
    return records


def read_prompt_file(path: Path) -> List[PromptRecord]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return records_from_jsonl(path)
    if suffix == ".csv":
        return records_from_csv(path)
    if suffix == ".json":
        return records_from_json(path)
    return records_from_text(path)


def apply_templates(records: Sequence[PromptRecord], templates: Sequence[str]) -> List[PromptRecord]:
    if not templates:
        return list(records)

    templated: List[PromptRecord] = []
    for record in records:
        for template_idx, template in enumerate(templates):
            if "CLASSNAME" in template:
                text = template.replace("CLASSNAME", record.prompt)
            elif "{}" in template:
                text = template.format(record.prompt)
            else:
                text = f"{template} {record.prompt}"
            metadata = dict(record.metadata)
            metadata["source_prompt"] = record.prompt
            metadata["applied_template"] = template
            metadata["applied_template_id"] = str(template_idx)
            templated.append(PromptRecord(prompt_id=f"{record.prompt_id}:t{template_idx}", prompt=text, metadata=metadata))
    return templated


def collect_prompts(args: argparse.Namespace) -> List[PromptRecord]:
    records: List[PromptRecord] = []
    for idx, prompt in enumerate(args.prompt or []):
        records.append(PromptRecord(prompt_id=str(idx), prompt=prompt))
    if args.prompts_file:
        records.extend(read_prompt_file(Path(args.prompts_file).expanduser().resolve()))
    records = apply_templates(records, args.template or [])

    if not records:
        raise ValueError("Provide at least one prompt with --prompt or --prompts-file")

    cleaned = [
        PromptRecord(prompt_id=record.prompt_id, prompt=record.prompt.strip(), metadata=record.metadata)
        for record in records
        if record.prompt.strip()
    ]
    if not cleaned:
        raise ValueError("All prompts were empty")
    return cleaned


def encode_prompts(
    records: Sequence[PromptRecord],
    model: torch.nn.Module,
    tokenizer,
    tokenize_fn,
    device: torch.device,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            end = min(start + batch_size, len(records))
            texts = [record.prompt for record in records[start:end]]
            token_ids = tokenize_fn(tokenizer, texts).to(device)
            encoded = model.encode_text(token_ids, normalize=normalize)
            embeddings.append(encoded.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(embeddings, axis=0)


def parse_ensemble_fields(value: str | None) -> List[str]:
    if value is None:
        return []
    fields = [field.strip() for field in value.split(",") if field.strip()]
    if not fields:
        raise ValueError("--ensemble-by must contain at least one metadata field")
    return fields


def record_field(record: PromptRecord, field_name: str) -> str:
    if field_name == "prompt_id":
        return record.prompt_id
    if field_name == "prompt":
        return record.prompt
    return record.metadata.get(field_name, "")


def validate_ensemble_fields(records: Sequence[PromptRecord], fields: Sequence[str]) -> None:
    missing = [
        field
        for field in fields
        if field not in {"prompt_id", "prompt"} and not any(field in record.metadata for record in records)
    ]
    if missing:
        raise ValueError(
            f"--ensemble-by field(s) not found in prompt metadata: {missing}. "
            "For prompt_gen.py outputs, use polarity,category,concept_id."
        )


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def ensemble_embeddings(
    records: Sequence[PromptRecord],
    embeddings: np.ndarray,
    fields: Sequence[str],
    normalize: bool,
) -> tuple[List[PromptRecord], np.ndarray]:
    validate_ensemble_fields(records, fields)

    groups: OrderedDict[tuple[str, ...], List[int]] = OrderedDict()
    for idx, record in enumerate(records):
        key = tuple(record_field(record, field) for field in fields)
        groups.setdefault(key, []).append(idx)

    ensemble_records: List[PromptRecord] = []
    ensemble_vectors: List[np.ndarray] = []
    for key, indices in groups.items():
        group_records = [records[idx] for idx in indices]
        group_embeddings = embeddings[indices]
        vector = group_embeddings.mean(axis=0, dtype=np.float32)
        if normalize:
            vector = l2_normalize_rows(vector.reshape(1, -1))[0]
        vector = vector.astype(np.float32, copy=False)
        ensemble_vectors.append(vector)

        metadata: Dict[str, str] = {field: value for field, value in zip(fields, key)}
        for candidate in ("concept", "concept_id", "category", "polarity"):
            if candidate not in metadata:
                values = {record.metadata.get(candidate, "") for record in group_records}
                if len(values) == 1:
                    metadata[candidate] = values.pop()
        metadata["ensemble_size"] = str(len(indices))
        metadata["ensemble_prompt_ids"] = json.dumps([record.prompt_id for record in group_records], ensure_ascii=False)
        metadata["ensemble_prompts"] = json.dumps([record.prompt for record in group_records], ensure_ascii=False)

        prompt_id = "__".join(f"{field}={value}" for field, value in zip(fields, key))
        if len(fields) == 1:
            prompt_label = key[0]
        else:
            prompt_label = " | ".join(f"{field}: {value}" for field, value in zip(fields, key))
        ensemble_records.append(PromptRecord(prompt_id=prompt_id, prompt=prompt_label, metadata=metadata))

    return ensemble_records, np.stack(ensemble_vectors, axis=0)


def write_h5(
    output_path: Path,
    records: Sequence[PromptRecord],
    embeddings: np.ndarray,
    checkpoint_path: str,
    checkpoint_sha256: str,
    normalize: bool,
    overwrite: bool,
    ensemble_fields: Sequence[str],
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    string_dtype = h5py.string_dtype(encoding="utf-8")
    prompt_ids = np.asarray([record.prompt_id for record in records], dtype=object)
    prompts = np.asarray([record.prompt for record in records], dtype=object)

    with h5py.File(tmp_path, "w") as h5:
        features = h5.create_dataset("features", data=embeddings.astype(np.float32, copy=False))
        features.attrs["embedding_space"] = "conch_v1_text_aligned"
        features.attrs["normalized"] = bool(normalize)
        contract = embedding_contract(checkpoint_sha256)
        for key, value in contract.items():
            h5.attrs[key] = value
            features.attrs[key] = value
        h5.create_dataset("prompt_ids", data=prompt_ids, dtype=string_dtype)
        h5.create_dataset("prompts", data=prompts, dtype=string_dtype)
        write_metadata_group(h5, records, string_dtype)
        h5.attrs["model"] = "conch_ViT-B-16"
        h5.attrs["checkpoint_path"] = checkpoint_path
        h5.attrs["embedding_type"] = "text"
        h5.attrs["normalized"] = bool(normalize)
        h5.attrs["ensemble"] = bool(ensemble_fields)
        h5.attrs["ensemble_by"] = ",".join(ensemble_fields)

    tmp_path.replace(output_path)


def write_metadata_group(h5: h5py.File, records: Sequence[PromptRecord], string_dtype) -> None:
    metadata_keys = sorted({key for record in records for key in record.metadata})
    if not metadata_keys:
        return

    group = h5.create_group("metadata")
    preferred = [key for key in PROMPT_GEN_METADATA_FIELDS if key in metadata_keys]
    remaining = [key for key in metadata_keys if key not in preferred]
    for key in preferred + remaining:
        values = np.asarray([record.metadata.get(key, "") for record in records], dtype=object)
        group.create_dataset(key, data=values, dtype=string_dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode text prompts with the CONCH v1 text encoder for image-text retrieval."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Text prompt to encode. Can be provided multiple times.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help=(
            "Prompt file. Supports prompt_gen.py outputs: prompts.jsonl, prompts.csv, "
            "and prompts.txt. .json accepts a list of strings, a list of {id,prompt} "
            "objects, or an id->prompt object."
        ),
    )
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        help=(
            "Optional prompt template. Use '{}' or CLASSNAME as the placeholder. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument("--output", required=True, help="Output H5 path.")
    parser.add_argument("--batch-size", type=int, default=256, help="Prompts encoded per batch.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--checkpoint-path",
        default="hf_hub:MahmoodLab/conch",
        help=(
            "CONCH v1 checkpoint file/directory or hf_hub:OWNER/REPO@REVISION. "
            "The resolved checkpoint bytes are SHA-256 fingerprinted."
        ),
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize text embeddings. Leave unset for retrieval.",
    )
    parser.add_argument(
        "--ensemble-by",
        default=None,
        help=(
            "Comma-separated fields used to average prompt embeddings into ensemble embeddings. "
            "For prompt_gen.py outputs, use polarity,category,concept_id."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    records = collect_prompts(args)
    ensemble_fields = parse_ensemble_fields(args.ensemble_by)
    device = choose_device(args.device)
    checkpoint = resolve_conch_checkpoint(args.checkpoint_path)
    model, tokenizer, tokenize_fn = load_conch_model_and_tokenizer(str(checkpoint.path), device)
    normalize = not args.no_normalize

    embeddings = encode_prompts(
        records=records,
        model=model,
        tokenizer=tokenizer,
        tokenize_fn=tokenize_fn,
        device=device,
        batch_size=args.batch_size,
        normalize=normalize,
    )
    output_records = records
    output_embeddings = embeddings
    if ensemble_fields:
        output_records, output_embeddings = ensemble_embeddings(
            records,
            embeddings,
            ensemble_fields,
            normalize=normalize,
        )
    write_h5(
        output_path=Path(args.output).expanduser().resolve(),
        records=output_records,
        embeddings=output_embeddings,
        checkpoint_path=args.checkpoint_path,
        checkpoint_sha256=checkpoint.sha256,
        normalize=normalize,
        overwrite=args.overwrite,
        ensemble_fields=ensemble_fields,
    )
    if ensemble_fields:
        print(
            f"Encoded {len(records)} prompt(s), ensembled into {len(output_records)} group(s) "
            f"by {','.join(ensemble_fields)} to {args.output}"
        )
    else:
        print(f"Encoded {len(records)} prompt(s) to {args.output}")


if __name__ == "__main__":
    main()

# python encode_conch_text_prompts.py --prompts-file prompt_bank/prompts.jsonl --output conch_text_concept_ensemble.h5 --ensemble-by polarity,category,concept_id --device cuda:0 --checkpoint-path __REPATH_PRIVATE_PROJECT_ROOT_002__/path_ad/backbones/weights/conch/pytorch_model.bin
