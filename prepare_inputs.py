#!/usr/bin/env python
"""Prepare train/validation/scaler files from user-supplied exact upstream inputs.

No downloads, reserved-partition exports, old mask banks or window generation.
The upstream USHCN legacy pickle is read only after its pinned SHA-256 matches.
This portable preparation path is source-traced and synthetic-tested, not a
bit-identical reconstruction verified against the historical real payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from preprocessing import load_jena_weather, prepare_ushcn, prepare_weather

SPEC = json.loads((Path(__file__).parent / "preprocessing/source_spec.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(path: Path, source: str) -> None:
    if sha256(path) != SPEC["sources"][source]["sha256"]:
        raise ValueError(f"SHA-256 mismatch for pinned source {source}; no inputs prepared")


def load_pinned_ushcn(path: Path):
    # Hash and deserialize the same open stream, never an unchecked pickle.
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != SPEC["sources"]["ushcn_processed"]["sha256"]:
            raise ValueError("SHA-256 mismatch for pinned source ushcn_processed")
        stream.seek(0)
        records = torch.load(stream, map_location="cpu", weights_only=False)
    if not isinstance(records, (list, tuple)):
        raise ValueError("USHCN source must contain a record sequence")
    return records


def isolated_record(record):
    """Do not serialize tensor views retaining unselected source storage."""
    record_id, *tensors = record
    if isinstance(record_id, np.integer):
        record_id = int(record_id)
    if isinstance(record_id, bool) or not isinstance(record_id, (str, int)) or not str(record_id):
        raise ValueError("record_id must be a nonempty string or integer")
    return (record_id, *(tensor.contiguous().clone() for tensor in tensors))


def write_dataset(output: Path, dataset: str, splits, scaler, source_keys):
    if set(splits) != {"train", "val"} or dataset not in {"weather", "ushcn"}:
        raise ValueError("only Weather/USHCN training and validation outputs are permitted")
    portable = {
        split: isolated_record(value) if dataset == "weather" else [isolated_record(record) for record in value]
        for split, value in splits.items()
    }
    directory = output / dataset
    directory.mkdir(parents=True, exist_ok=False)
    scaler.save(directory / "scaler.npz")
    for split, payload in portable.items():
        torch.save(payload, directory / f"{split}.pt")
    receipt = {
        "dataset": dataset,
        "scope": "regenerated train/validation inputs; no reserved-partition outputs",
        "sources": {key: SPEC["sources"][key] for key in source_keys},
        "files": {name: sha256(directory / name) for name in ("train.pt", "val.pt", "scaler.npz")},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "historical_byte_identity_verified": False,
        "serialization_caveat": SPEC["serialization_caveat"],
    }
    (directory / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="dataset", required=True)
    weather = commands.add_parser("weather2020", help="use the two exact official 2020 ZIPs")
    weather.add_argument("--weather-a", type=Path, required=True)
    weather.add_argument("--weather-b", type=Path, required=True)
    weather.add_argument("--out", type=Path, required=True, help="parent of a new weather/ directory")
    ushcn = commands.add_parser("ushcn", help="use the exact pinned processed USHCN tensor")
    ushcn.add_argument("--ushcn", type=Path, required=True)
    ushcn.add_argument("--out", type=Path, required=True, help="parent of a new ushcn/ directory")
    args = parser.parse_args()
    dataset = "weather" if args.dataset == "weather2020" else "ushcn"
    if (args.out / dataset).exists():
        parser.error("dataset output directory already exists; nothing will be overwritten")
    if dataset == "weather":
        keys = ("weather_2020a", "weather_2020b")
        verify_source(args.weather_a, keys[0])
        verify_source(args.weather_b, keys[1])
        splits, scaler = prepare_weather(load_jena_weather([args.weather_a, args.weather_b]))
    else:
        keys = ("ushcn_processed",)
        splits, scaler = prepare_ushcn(load_pinned_ushcn(args.ushcn))
    receipt = write_dataset(args.out, dataset, splits, scaler, keys)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
