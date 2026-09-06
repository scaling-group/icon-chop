"""Merge MFC HDF5 chunks with validation and no implicit overwrite."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py


def chunk_number(path: Path) -> int:
    match = re.search(r"_(\d+)\.h5$", path.name)
    if match is None:
        raise ValueError(f"Not a numbered HDF5 chunk: {path}")
    return int(match.group(1))


def merge(
    input_dir: Path,
    prefix: str,
    output_path: Path,
    *,
    expected_groups: int | None,
    delete_sources: bool,
) -> None:
    sources = sorted(input_dir.glob(f"{prefix}_[0-9]*.h5"), key=chunk_number)
    if not sources:
        raise FileNotFoundError(f"No chunks matching {prefix}_[0-9]*.h5 in {input_dir}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"Refusing to overwrite partial file {partial}")

    copied = 0
    try:
        with h5py.File(partial, "w") as destination:
            for source_path in sources:
                with h5py.File(source_path, "r") as source:
                    for group_name in source:
                        if group_name in destination:
                            raise ValueError(f"Duplicate group {group_name}")
                        source.copy(group_name, destination)
                        copied += 1
        if expected_groups is not None and copied != expected_groups:
            raise ValueError(f"Merged {copied} groups; expected {expected_groups}")
        partial.rename(output_path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise

    if delete_sources:
        for source_path in sources:
            source_path.unlink()
    print(f"Merged {len(sources)} chunk(s), {copied} groups -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int)
    parser.add_argument("--delete-sources", action="store_true")
    args = parser.parse_args()
    merge(
        args.input_dir.expanduser().resolve(),
        args.prefix,
        args.output.expanduser().resolve(),
        expected_groups=args.expected_groups,
        delete_sources=args.delete_sources,
    )


if __name__ == "__main__":
    main()
