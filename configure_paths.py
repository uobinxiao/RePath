#!/usr/bin/env python3
"""Configure the local paths and SLURM settings in the released source files.

1. Fill in REPLACEMENTS below. Leave optional or unused values empty.
2. Preview with: python3 configure_paths.py --dry-run
3. Apply with:   python3 configure_paths.py

Path values must be absolute POSIX paths without whitespace, quotes, or shell
metacharacters. Existing suffixes are preserved: fill in each root directory,
not the full path to a model, data file, or environment activation script.

Uses Python's standard library, Git, and helpers from anonymize.py. The private
author recovery mapping is not needed. Replacement values are never printed.
"""

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from anonymize import atomic_write, working_files


# Fill in the values below. An empty string leaves the placeholder unchanged.
# Do not remove the placeholder keys or change their numbering.
REPLACEMENTS = {
    # Home directory containing libvips/bin, openslide/bin, libdicom/bin,
    # and py310_env/bin/activate. Adjust those suffixes separately if needed.
    "__REPATH_PRIVATE_HOME_001__": "",

    # Personal project root containing tcga-extra*, path_ad/backbones/weights,
    # logs, and dinov2_checkpoints.
    "__REPATH_PRIVATE_PROJECT_ROOT_002__": "",

    # Shared data root containing TCGA/svs_files and TCGA_processed*.
    "__REPATH_PRIVATE_PROJECT_ROOT_003__": "",

    # Output root for tcga_conch_v1, prompt_bank, and prompt embeddings.
    "__REPATH_PRIVATE_SCRATCH_ROOT_004__": "",

    # Optional alternative output root used only in a compilation example.
    # The example appends tcga_conch_v1/offline_index to this root.
    "__REPATH_PRIVATE_SCRATCH_ROOT_005__": "",

    # Directory containing cached whole-slide image files.
    "__REPATH_PRIVATE_SCRATCH_ROOT_006__": "",

    # Processed data root containing the 5x, 10x, 20x, and 40x directories
    # named <MAG>x_256px_0px_overlap, including features and HSV filters.
    "__REPATH_PRIVATE_SCRATCH_ROOT_007__": "",

    # WSI root containing the slide files, as used by the offline JSON template.
    "__REPATH_PRIVATE_SCRATCH_ROOT_008__": "",

    # Optional slide filename used only in a commented example; include .svs.
    "__REPATH_PRIVATE_SLIDE_009__": "",

    # SLURM partition for the tissue-patching job.
    "__REPATH_PRIVATE_SLURM_PARTITION_010__": "",

    # SLURM partition for segmentation, feature extraction, clustering,
    # and pretraining. It may be the same as the partition above.
    "__REPATH_PRIVATE_SLURM_PARTITION_011__": "",

    # SLURM allocation/account for the submitted jobs.
    "__REPATH_PRIVATE_SLURM_ACCOUNT_012__": "",
}


TOKEN_RE = re.compile(r"__REPATH_PRIVATE_[A-Z_]+_[0-9]{3,}__")
UNSAFE_CHARACTERS = set("\"'\\$`;&|<>*?(){}[]!#")


def configured_values():
    """Validate settings before changing any source files."""
    configured = {}
    for token, value in REPLACEMENTS.items():
        if not isinstance(token, str) or not TOKEN_RE.fullmatch(token):
            raise ValueError("REPLACEMENTS contains an invalid placeholder key")
        if not isinstance(value, str):
            raise ValueError(f"The value for {token} must be a string")
        if value == "":
            continue
        if "__REPATH_PRIVATE_" in value:
            raise ValueError(f"Replace {token} with a real value, not another placeholder")
        if any(ord(character) < 32 or ord(character) == 127 or character.isspace()
               or character in UNSAFE_CHARACTERS for character in value):
            raise ValueError(f"The value for {token} contains whitespace, quotes, or shell metacharacters")
        if any(kind in token for kind in ("_HOME_", "_PROJECT_ROOT_", "_SCRATCH_ROOT_")):
            if not value.startswith("/"):
                raise ValueError(f"The value for {token} must be an absolute POSIX path")
        if "_SLURM_" in token and not re.fullmatch(r"[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*", value):
            raise ValueError(f"The value for {token} must be a SLURM identifier")
        configured[token] = value
    return configured


def configure(root, dry_run=False):
    configured = configured_values()
    pattern = re.compile("|".join(re.escape(token) for token in configured)) if configured else None
    updates, remaining = [], Counter()
    replacement_count = 0

    for path in working_files(root):
        # Keep documentation examples and all setup/recovery tools unchanged.
        if path.name.lower().startswith("readme"):
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        changed, count = pattern.subn(lambda match: configured[match.group()], text) if pattern else (text, 0)
        remaining.update(TOKEN_RE.findall(changed))
        if count:
            updates.append((path, raw, changed.encode("utf-8"), count))
            replacement_count += count

    if not configured:
        print("No values configured. Edit REPLACEMENTS at the top of configure_paths.py.")
    for path, _, _, count in updates:
        print(f"{'Would update' if dry_run else 'Update'} {path.relative_to(root).as_posix()}: {count} replacements")
    print(f"{'Preview' if dry_run else 'Apply'}: {len(updates)} files, {replacement_count} replacements.")

    if not dry_run:
        # Preflight all files before writing. Atomic writes preserve permissions
        # and newline bytes; only the configured placeholders are changed.
        for path, original, _, _ in updates:
            if path.is_symlink() or path.resolve() != path.absolute() or path.read_bytes() != original:
                raise ValueError("File changed during configuration: " + path.relative_to(root).as_posix())
        for path, original, changed, _ in updates:
            if path.is_symlink() or path.resolve() != path.absolute() or path.read_bytes() != original:
                raise ValueError("File changed during configuration: " + path.relative_to(root).as_posix())
            atomic_write(path, changed)

    if remaining:
        print("Remaining placeholders (empty values and unused examples may be intentional):")
        for token, count in sorted(remaining.items()):
            note = " - add this key to REPLACEMENTS" if token not in REPLACEMENTS else ""
            print(f"  {token}: {count} occurrences{note}")
    else:
        print("No placeholders remain in the scanned source/configuration files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project Git root (defaults to this script's directory)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying files")
    args = parser.parse_args()
    try:
        configure(args.root.resolve(), args.dry_run)
    except ValueError as error:
        parser.exit(1, f"Configuration stopped: {error}.\n")
    except (OSError, subprocess.CalledProcessError):
        parser.exit(1, "Configuration failed while reading/writing files or running Git.\n")


if __name__ == "__main__":
    main()
