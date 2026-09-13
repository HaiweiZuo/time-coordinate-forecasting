"""Verify checkpoint ZIP hashes and every member without extraction or unpickling."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(directory, root):
    release = json.loads((root / "release_assets.json").read_text(encoding="utf-8"))
    registry = json.loads((root / "checkpoint_index.json").read_text(encoding="utf-8"))
    records = registry["checkpoints"]
    expected = {row["checkpoint"]: row for row in records}
    if len(expected) != 48 or len(records) != 48:
        raise ValueError("registry must identify 48 distinct checkpoints")
    seen = set()
    for asset in release["assets"]:
        path = directory / asset["name"]
        if path.name != asset["name"]:
            raise ValueError("asset names must be basenames")
        with path.open("rb") as stream:
            if path.stat().st_size != asset["bytes"] or digest(stream) != asset["sha256"]:
                raise ValueError(f"archive hash or size mismatch: {path.name}")
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                members = archive.infolist()
                if len(members) != asset["checkpoints"]:
                    raise ValueError(f"unexpected member count: {path.name}")
                for member in members:
                    name = member.filename
                    if name not in expected or name in seen or member.is_dir():
                        raise ValueError(f"unexpected or duplicate checkpoint: {name}")
                    row = expected[name]
                    with archive.open(member) as payload:
                        if member.file_size != row["bytes"] or digest(payload) != row["sha256"]:
                            raise ValueError(f"checkpoint hash or size mismatch: {name}")
                    seen.add(name)
        print(f"Verified {path.name} ({len(members)} checkpoints)")
    if seen != set(expected):
        raise ValueError("archives do not cover the complete checkpoint registry")
    print("Verified all 48 checkpoint files; no files extracted or deserialized.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    args = parser.parse_args()
    verify(args.archives, Path(__file__).resolve().parent)

