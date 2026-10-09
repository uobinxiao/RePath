#!/usr/bin/env python3
"""Replace private paths and cluster identifiers with reversible placeholders.

Preview: python anonymize.py --dry-run
Apply:   python anonymize.py
Restore: python restore_anonymization.py

Uses only the Python standard library and Git. Original values are discovered
from the working tree, never embedded in these scripts. Ignored data, binary
files, runtime logs, Git history, and commit identities are not rewritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


STATE_DIR = ".anonymize-private"
MAP_NAME = "mapping.json"
MARKER = "__REPATH_PRIVATE_"
TEXT_SUFFIXES = {
    ".py", ".sh", ".yaml", ".yml", ".json", ".jsonl", ".md", ".txt",
    ".toml", ".ini", ".cfg", ".conf", ".sbatch",
}
SKIP_DIRS = {
    ".git", ".agents", ".codex", ".pytest_cache", "__pycache__",
    ".venv", "venv", "node_modules", STATE_DIR,
}
EXAMPLE_NAMES = {"user", "username", "you", "example", "placeholder"}
COMPONENT = r"[A-Za-z0-9_.-]+"
PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_/])(?:"
    rf"/cluster/projects/{COMPONENT}/{COMPONENT}|"
    rf"/cluster/home/{COMPONENT}|"
    rf"/(?:Users|home|scratch)/{COMPONENT})"
)
ASSIGNMENT_RE = re.compile(
    r"(?m)^\s*(?:export\s+)?SLURM_(PARTITION|ACCOUNT)\s*=\s*"
    r"[\"']?([A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*)[\"']?\s*(?:#.*)?$"
)
SBATCH_RE = re.compile(
    r"(?m)^\s*#SBATCH\s+(--partition|--account|-p|-A)"
    r"(?:=|\s+)([A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*)"
)
SLIDE_RE = re.compile(
    r"\bTCGA-[A-Z0-9]{2}-[A-Z0-9]{4}"
    r"(?:[A-Za-z0-9_.-]*[A-Za-z0-9])?(?![A-Za-z0-9_.-])"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=True
    )
    return result.stdout


def working_files(root):
    """Include current tracked/untracked source, including nested repositories."""
    if Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).strip()).resolve() != root:
        raise ValueError("--root must be the top level of the project Git repository")
    files = set()

    def collect(repo):
        names = git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
        for name in names.split(b"\0"):
            if not name:
                continue
            path = repo / os.fsdecode(name)
            relative = path.relative_to(root)
            if any(part in SKIP_DIRS for part in relative.parts):
                continue
            if path.is_symlink():
                continue
            if path.is_dir() and (path / ".git").exists():
                collect(path)
            elif path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                if path.resolve() != path.absolute():
                    continue  # Do not follow a symlinked parent directory.
                if path.name not in {"anonymize.py", "restore_anonymization.py", "configure_paths.py"}:
                    files.add(path)

    collect(root)
    # Nested Git repositories have their own ignore rules; also honor the
    # enclosing project's rules (for example metadata*.json).
    if files:
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--no-index", "-z", "--stdin"],
            input=b"".join(os.fsencode(path.relative_to(root).as_posix()) + b"\0" for path in files),
            capture_output=True,
        )
        if result.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(result.returncode, result.args)
        ignored = {root / os.fsdecode(name) for name in result.stdout.split(b"\0") if name}
        files.difference_update(ignored)
    return sorted(files)


def private_values(texts):
    values = {}
    for text in texts:
        for match in PATH_RE.finditer(text):
            value = match.group()
            if value.rsplit("/", 1)[-1].lower() in EXAMPLE_NAMES:
                continue
            if value.startswith("/cluster/projects/"):
                kind = "PROJECT_ROOT"
            elif value.startswith("/scratch/"):
                kind = "SCRATCH_ROOT"
            else:
                kind = "HOME"
            values[value] = kind
        for match in ASSIGNMENT_RE.finditer(text):
            value = match.group(2)
            if value.lower() not in EXAMPLE_NAMES and not value.startswith("__"):
                values[value] = "SLURM_" + match.group(1).upper()
        for match in SBATCH_RE.finditer(text):
            value = match.group(2)
            if value.lower() not in EXAMPLE_NAMES and not value.startswith("__"):
                kind = "PARTITION" if match.group(1) in {"--partition", "-p"} else "ACCOUNT"
                values[value] = "SLURM_" + kind
        for match in SLIDE_RE.finditer(text):
            values[match.group()] = "SLIDE"
    return values


def replacement_pattern(values):
    # Boundaries avoid changing paths/identifiers which merely share a prefix.
    alternatives = []
    for value in sorted(values, key=lambda item: (-len(item), item)):
        left = r"(?<![A-Za-z0-9_/])" if value.startswith("/") else r"(?<![A-Za-z0-9_.-])"
        alternatives.append(left + re.escape(value) + r"(?![A-Za-z0-9_.-])")
    return re.compile("|".join(alternatives)) if alternatives else None


def replace_text(text, values, tokens, pattern):
    # Account/partition names are replaced only in their settings, never in
    # unrelated words or Python identifiers (a partition may be named "gpu").
    spans = []
    if pattern is not None:
        spans.extend((match.start(), match.end(), match.group()) for match in pattern.finditer(text))
    for regex in (ASSIGNMENT_RE, SBATCH_RE):
        for match in regex.finditer(text):
            value = match.group(2)
            if value in values and values[value].startswith("SLURM_"):
                spans.append((match.start(2), match.end(2), value))
    counts, parts, end = {}, [], 0
    for start, stop, value in sorted(spans):
        if start < end:
            raise ValueError("Overlapping anonymization rules")
        parts.extend((text[end:start], tokens[value]))
        counts[value] = counts.get(value, 0) + 1
        end = stop
    parts.append(text[end:])
    return "".join(parts).encode("utf-8"), counts


def atomic_write(path, data, mode=None):
    """Replace one file atomically, preserving its permissions."""
    if mode is None:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, name = tempfile.mkstemp(prefix=".anonymize-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_mapping(path, mapping):
    payload = json.dumps(mapping, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    atomic_write(path, payload, 0o600)


def load_mapping(path):
    mapping = json.loads(path.read_bytes())
    if not isinstance(mapping, dict) or mapping.get("version") != 1 or mapping.get("status") not in {"active", "restored"}:
        raise ValueError("Unrecognized anonymization mapping")
    if not isinstance(mapping.get("files"), list):
        raise ValueError("Invalid anonymization mapping")
    seen = set()
    for record in mapping["files"]:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Invalid file record in the mapping")
        if record["path"] in seen:
            raise ValueError("Duplicate file record in the mapping")
        seen.add(record["path"])
        for key in ("before_sha256", "after_sha256"):
            if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", record[key]):
                raise ValueError("Invalid file checksum in the mapping")
        items = record.get("replacements")
        if not isinstance(items, list) or not items:
            raise ValueError("Missing replacement records in the mapping")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("placeholder"), str):
                raise ValueError("Invalid placeholder record in the mapping")
            if not re.fullmatch(r"__REPATH_PRIVATE_[A-Z_]+_[0-9]{3,}__", item["placeholder"]):
                raise ValueError("Invalid placeholder in the mapping")
            if not isinstance(item.get("original"), str) or MARKER in item["original"]:
                raise ValueError("Invalid original value in the mapping")
            if type(item.get("count")) is not int or item["count"] < 1:
                raise ValueError("Invalid placeholder count in the mapping")
    return mapping


def mapped_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Invalid file path in the mapping")
    path = root / relative
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError("Mapped file is missing or has become a symlink: " + name)
    return path


def apply(root, dry_run=False):
    map_path = root / STATE_DIR / MAP_NAME
    if map_path.is_symlink() or map_path.parent.is_symlink():
        raise ValueError("The private mapping location must not be a symlink")
    if map_path.exists():
        existing = load_mapping(map_path)
        if existing["status"] == "active":
            for record in existing["files"]:
                path = mapped_path(root, record["path"])
                if digest(path.read_bytes()) != record["after_sha256"]:
                    raise ValueError("An active mapping exists; restore before anonymizing again")
            print("Already anonymized; no files changed. Restore before scanning new files.")
            return

    sources = {}
    for path in working_files(root):
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if MARKER in text:
            raise ValueError("Reserved placeholder already exists: " + str(path.relative_to(root)))
        sources[path] = (raw, text)

    values = private_values(text for _, text in sources.values())
    if not values:
        print("No matching private values found; no files changed.")
        return
    tokens = {
        value: f"{MARKER}{values[value]}_{index:03d}__"
        for index, value in enumerate(sorted(values), 1)
    }
    pattern = replacement_pattern({value: kind for value, kind in values.items()
                                   if not kind.startswith("SLURM_")})
    records, updates = [], []
    for path, (raw, text) in sources.items():
        changed, counts = replace_text(text, values, tokens, pattern)
        if changed == raw:
            continue
        records.append({
            "path": path.relative_to(root).as_posix(),
            "before_sha256": digest(raw),
            "after_sha256": digest(changed),
            "replacements": [
                {"placeholder": tokens[value], "original": value, "count": count}
                for value, count in sorted(counts.items())
            ],
        })
        updates.append((path, raw, changed))

    for record in records:
        print(f"{record['path']}: {sum(item['count'] for item in record['replacements'])} replacements")
    print(f"{'Preview' if dry_run else 'Apply'}: {len(records)} files; original values are not printed.")
    if dry_run or not records:
        return
    ignored = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--no-index", "--quiet", "--",
         f"{STATE_DIR}/{MAP_NAME}"], capture_output=True
    )
    if ignored.returncode != 0:
        raise ValueError(f"Add /{STATE_DIR}/ to .gitignore before applying")
    if git(root, "ls-files", "--", STATE_DIR):
        raise ValueError("The private mapping directory must not contain tracked files")
    # Save recovery information BEFORE any source file is changed.
    map_path.parent.mkdir(mode=0o700, exist_ok=True)
    save_mapping(map_path, {"version": 1, "status": "active", "files": records})
    for path, original, changed in updates:
        if path.is_symlink() or path.resolve() != path.absolute() or path.read_bytes() != original:
            raise ValueError("File changed during anonymization; use restore: " + str(path.relative_to(root)))
        atomic_write(path, changed)
    print(f"Private recovery mapping: {STATE_DIR}/{MAP_NAME}. Keep it locally until restored.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project Git root (defaults to the directory containing this script)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying files")
    args = parser.parse_args()
    try:
        apply(args.root.resolve(), args.dry_run)
    except ValueError as error:
        parser.exit(1, f"Anonymization stopped: {error}.\n")
    except (OSError, KeyError, subprocess.CalledProcessError):
        parser.exit(1, "Anonymization failed while reading/writing files or running Git. "
                    "If a mapping exists, use restore_anonymization.py before retrying.\n")


if __name__ == "__main__":
    main()
