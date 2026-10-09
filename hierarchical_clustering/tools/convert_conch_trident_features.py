import argparse
import os
import tempfile
from pathlib import Path
from typing import List
import h5py
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .conch_checkpoint import embedding_contract, resolve_conch_checkpoint
except ImportError:
    from conch_checkpoint import embedding_contract, resolve_conch_checkpoint

def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def discover_inputs(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    paths = sorted(path for path in input_path.rglob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {input_path}")
    return paths


def default_output_path(input_path: Path, output_root: Path | None, base_input: Path, suffix: str) -> Path:
    if output_root is not None:
        if base_input.is_file():
            return output_root
        return output_root / input_path.relative_to(base_input)
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def resolve_output_root(input_path: Path, output_arg: Path | None, suffix: str) -> Path | None:
    if input_path.is_file():
        return output_arg
    output_root = output_arg or input_path.with_name(f"{input_path.name}{suffix}")
    resolved_input = input_path.resolve()
    resolved_output = output_root.resolve()
    if resolved_output == resolved_input or resolved_input in resolved_output.parents:
        raise ValueError(
            "Directory output must be separate from, and not nested under, the raw input directory."
        )
    return resolved_output


def copy_h5_without_features(src: h5py.File, dst: h5py.File, features_key: str) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key in src.keys():
        if key == features_key:
            continue
        src.copy(key, dst, name=key)


def normalize_projected_batch(
    projected: torch.Tensor,
    source_path: Path,
    row_start: int,
) -> torch.Tensor:
    """Normalize a projected batch and enforce the H5 normalized=True contract."""
    if projected.ndim != 2:
        raise ValueError(
            f"{source_path}: projected features must be 2-D, got {tuple(projected.shape)}"
        )
    projected = projected.float()
    finite_rows = torch.isfinite(projected).all(dim=1)
    norms = torch.linalg.vector_norm(projected, dim=-1)
    valid_rows = finite_rows & torch.isfinite(norms) & (norms > 1e-12)
    if not bool(valid_rows.all()):
        local_row = int(torch.nonzero(~valid_rows, as_tuple=False)[0].item())
        norm = float(norms[local_row].detach().cpu())
        raise ValueError(
            f"{source_path}: projected feature at row {row_start + local_row} is "
            f"non-finite or near-zero before normalization (norm={norm:.8g}); "
            "regenerate the raw CONCH feature if this persists"
        )
    normalized = F.normalize(projected, dim=-1)
    normalized_norms = torch.linalg.vector_norm(normalized, dim=-1)
    close_to_one = torch.isclose(normalized_norms, torch.ones_like(normalized_norms), atol=1e-5)
    if not bool(close_to_one.all()):
        local_row = int(torch.nonzero(~close_to_one, as_tuple=False)[0].item())
        raise ValueError(
            f"{source_path}: projected feature at row {row_start + local_row} failed "
            f"L2 normalization (norm={float(normalized_norms[local_row].detach().cpu()):.8g})"
        )
    return normalized


def write_projected_h5(
    src_path: Path,
    tmp_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    features_key: str,
    compression: str | None,
    checkpoint_sha256: str,
    checkpoint_source: str | None,
) -> None:
    with h5py.File(src_path, "r") as src:
        if features_key not in src:
            raise KeyError(f"{src_path}: missing dataset {features_key!r}")
        features = src[features_key]
        if features.ndim != 2:
            raise ValueError(f"{src_path}: {features_key!r} must be 2-D, got {features.shape}")
        embedding_space = features.attrs.get("embedding_space")
        if isinstance(embedding_space, bytes):
            embedding_space = embedding_space.decode("utf-8")
        if embedding_space == "conch_v1_text_aligned":
            raise ValueError(f"{src_path}: input features are already CONCH text-aligned")

        n_rows, feature_dim = features.shape
        projection_dim = int(model.visual.proj_contrast.shape[0])
        if feature_dim != projection_dim:
            raise ValueError(
                f"{src_path}: expected CONCH v1 Trident features with dim {projection_dim}, "
                f"got {feature_dim}. This script is only for Trident default conch_v1 features."
            )

        with h5py.File(tmp_path, "w") as dst:
            copy_h5_without_features(src, dst, features_key)
            contract = embedding_contract(checkpoint_sha256)
            for key, value in contract.items():
                dst.attrs[key] = value
            if checkpoint_source is not None:
                dst.attrs["checkpoint_source"] = checkpoint_source
            dataset_options = {
                "shape": (n_rows, projection_dim),
                "dtype": "float32",
                "compression": compression,
            }
            if n_rows == 0:
                dataset_options.update(
                    maxshape=(None, projection_dim),
                    chunks=(1, projection_dim),
                )
            else:
                dataset_options["chunks"] = (min(batch_size, n_rows), projection_dim)
            out = dst.create_dataset(features_key, **dataset_options)
            for key, value in features.attrs.items():
                out.attrs[key] = value
            out.attrs["embedding_space"] = "conch_v1_text_aligned"
            out.attrs["source_features"] = "trident_conch_v1_default_pre_projection"
            out.attrs["normalized"] = True
            for key, value in contract.items():
                out.attrs[key] = value

            with torch.inference_mode():
                for start in range(0, n_rows, batch_size):
                    end = min(start + batch_size, n_rows)
                    batch = torch.as_tensor(
                        np.asarray(features[start:end], dtype=np.float32),
                        device=device,
                    )
                    if not bool(torch.isfinite(batch).all()):
                        bad = torch.nonzero(~torch.isfinite(batch), as_tuple=False)[0]
                        raise ValueError(
                            f"{src_path}: raw feature is non-finite at row "
                            f"{start + int(bad[0].item())}, column {int(bad[1].item())}"
                        )
                    projected = model.visual.forward_project(batch)
                    projected = normalize_projected_batch(projected, src_path, start)
                    out[start:end] = projected.cpu().numpy().astype(np.float32, copy=False)

def convert_feature_dataset(
    src_path: Path,
    dst_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    features_key: str,
    compression: str | None,
    overwrite: bool,
    checkpoint_sha256: str,
    checkpoint_source: str | None = None,
) -> None:
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {dst_path}. Use --overwrite to replace it.")
    if src_path.resolve() == dst_path.resolve():
        raise ValueError("Refusing to overwrite the input file in place. Write to a different path.")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{dst_path.name}.",
        suffix=".tmp",
        dir=dst_path.parent,
    )
    os.close(file_descriptor)
    tmp_path = Path(temporary_name)
    try:
        write_projected_h5(
            src_path=src_path,
            tmp_path=tmp_path,
            model=model,
            device=device,
            batch_size=batch_size,
            features_key=features_key,
            compression=compression,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_source=checkpoint_source,
        )
        tmp_path.replace(dst_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_conch_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    try:
        from conch.open_clip_custom import create_model_from_pretrained
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
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Trident default conch_v1 H5 image features into CONCH v1 "
            "text-aligned embeddings for image-text retrieval."
        )
    )
    parser.add_argument("--input", required=True, help="Input H5 file or directory containing H5 files.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output H5 file when --input is a file, or output directory when --input is a directory. "
            "A file input defaults to a suffixed file; a directory input defaults to a separate "
            "sibling directory named INPUT_SUFFIX."
        ),
    )
    parser.add_argument("--pattern", default="*.h5", help="H5 glob pattern when --input is a directory.")
    parser.add_argument("--features-key", default="features", help="Name of the feature dataset in each H5 file.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Rows converted per batch.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--checkpoint-path",
        default="hf_hub:MahmoodLab/conch",
        help=(
            "CONCH v1 checkpoint file/directory or hf_hub:OWNER/REPO@REVISION. "
            "The resolved checkpoint bytes are SHA-256 fingerprinted."
        ),
    )
    parser.add_argument("--suffix", default="_text_aligned", help="Suffix used when --output is omitted.")
    parser.add_argument(
        "--compression",
        default=None,
        choices=["gzip", "lzf"],
        help="Optional H5 compression for the converted features dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    input_path = Path(args.input).expanduser().resolve()
    output_arg = Path(args.output).expanduser().resolve() if args.output else None
    output_root = resolve_output_root(input_path, output_arg, args.suffix)
    inputs = discover_inputs(input_path, args.pattern)
    device = choose_device(args.device)
    checkpoint = resolve_conch_checkpoint(args.checkpoint_path)
    model = load_conch_model(str(checkpoint.path), device)

    if input_path.is_file():
        outputs = [default_output_path(inputs[0], output_root, input_path, args.suffix)]
    else:
        outputs = [default_output_path(path, output_root, input_path, args.suffix) for path in inputs]

    for src_path, dst_path in zip(inputs, outputs):
        print(f"Converting {src_path} -> {dst_path}")
        convert_feature_dataset(
            src_path=src_path,
            dst_path=dst_path,
            model=model,
            device=device,
            batch_size=args.batch_size,
            features_key=args.features_key,
            compression=args.compression,
            overwrite=args.overwrite,
            checkpoint_sha256=checkpoint.sha256,
            checkpoint_source=checkpoint.source,
        )
    print(f"Converted {len(inputs)} file(s).")


if __name__ == "__main__":
    main()

# python convert_conch_trident_features.py --checkpoint-path __REPATH_PRIVATE_PROJECT_ROOT_002__/path_ad/backbones/weights/conch/pytorch_model.bin --input __REPATH_PRIVATE_SCRATCH_ROOT_007__/5x_256px_0px_overlap/features_conch_v1 --output __REPATH_PRIVATE_SCRATCH_ROOT_007__/5x_256px_0px_overlap/features_conch_v1_text_aligned --device cuda:0 --batch-size 8192
