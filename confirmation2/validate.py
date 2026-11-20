#!/usr/bin/env python
"""Health check of the single-archive reader on Weather 2023 (second protocol, section 4).

The two 2023 archives are merged into one temporary full-year archive. The
reader of confirmation2/prepare_data.py must then give the same tensors as
the two-archive reader of the first run and, after the unchanged 2020 scaler,
the Weather 2023 payload of the first run bit for bit. The 2023 records have
already been scored. No value is printed.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from confirmation.common import CONFIG, require, sha256, write_json

import numpy as np
import torch

from confirmation.prepare_data import header_of, read_year_archives
from confirmation2.prepare_data import read_archives, scaled_record

YEAR = 2023


def merge(archives: list[Path], target: Path) -> None:
    """One CSV: the header and rows of the first archive, then the rows of the others, bytes unchanged."""
    parts = []
    for number, path in enumerate(archives):
        with zipfile.ZipFile(path) as bundle:
            names = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
            require(len(names) == 1, "each Jena archive must contain exactly one CSV")
            lines = bundle.read(names[0]).splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            lines[-1] += b"\n"
        parts.extend(lines if number == 0 else lines[1:])
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(f"mpi_roof_{YEAR}.csv", b"".join(parts))


def same(left, right) -> bool:
    return left[0] == right[0] and all(a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(left[1:], right[1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-2023", type=Path, nargs=2, required=True, metavar=("A", "B"))
    parser.add_argument("--weather-2020a", type=Path, required=True)
    parser.add_argument("--original-data", type=Path, required=True)
    parser.add_argument("--first-run-payload", type=Path, required=True, help="weather2023.pt written by confirmation/prepare_data.py")
    parser.add_argument("--out", type=Path, required=True, help="JSON receipt")
    args = parser.parse_args()
    require(np.__version__ == CONFIG["evaluation"]["numpy_version"], "study preparation requires NumPy 1.26.4")
    torch.set_num_threads(1)
    expected = header_of(args.weather_2020a)[1:]
    archives = list(args.weather_2023)
    pair_raw, pair_metadata = read_year_archives(archives, expected, YEAR)
    with tempfile.TemporaryDirectory() as directory:
        merged = Path(directory) / f"mpi_roof_{YEAR}.zip"
        merge(archives, merged)
        single_raw, single_metadata = read_archives([merged], expected, YEAR)
    same_reader_pair, same_metadata_pair = read_archives(archives, expected, YEAR)
    stored = torch.load(args.first_run_payload, map_location="cpu", weights_only=True)
    scaled = scaled_record(single_raw, args.original_data / "weather" / "scaler.npz", f"weather:temporal{YEAR}")
    counts = [key for key in pair_metadata if key != "archives"]
    receipt = {
        "new_reader_on_the_two_archives_equals_first_reader": same(same_reader_pair, pair_raw) and same_metadata_pair == pair_metadata,
        "single_merged_archive_equals_two_archives": same(single_raw, pair_raw),
        "metadata_counts_equal": all(single_metadata[key] == pair_metadata[key] for key in counts),
        "raw_rows_equal": single_metadata["archives"][0]["raw_rows"] == sum(entry["raw_rows"] for entry in pair_metadata["archives"]),
        "scaled_payload_equals_first_run_payload_bitwise": same(scaled, stored),
        "first_run_payload_sha256": sha256(args.first_run_payload),
        "lattice_rows": single_metadata["lattice_rows"],
    }
    write_json(args.out, {"reader_2023_check": receipt})
    require(all(value is True for key, value in receipt.items() if key not in ("first_run_payload_sha256", "lattice_rows")),
            "single-archive reader does not reproduce the first run")
    print("PASS reader check on Weather 2023:", ", ".join(key for key, value in receipt.items() if value is True))


if __name__ == "__main__":
    main()
