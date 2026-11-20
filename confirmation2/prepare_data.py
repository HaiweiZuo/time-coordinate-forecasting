#!/usr/bin/env python
"""Prepare the Weather 2024 payload (second protocol, section 4). CPU only.

The script never prints, plots or summarises observation values. It reports
identities, counts and hashes only.

The source publishes 2024 as one full-year archive, so the two-archive reader
of the first run cannot be called. `read_archives` below repeats its row
semantics for any ordered list of archives of one year: Latin-1, strict
datetime, -9999 missing, mean of valid duplicates, complete ten-minute
lattice. confirmation2/validate.py shows on the 2023 archives that both
readers give the same tensors.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from confirmation.common import CONFIG, WEATHER_SCALER_SHA256, require, sha256, write_json
from confirmation2.common import (
    ARCHIVE, METADATA_FILE, PAYLOAD_FILE, PROTOCOL_SHA256, RECORD_ID, ROSTER, YEAR, build_roster, require_protocol,
)

import numpy as np
import torch

from confirmation.prepare_data import header_of, load_scaler
from confirmation.rosters import paired_across_arms
from prepare_inputs import isolated_record, verify_source

STEP = timedelta(minutes=10)


def read_archives(paths: list[Path], expected_columns: list[str], year: int):
    """Row semantics of confirmation.prepare_data.read_year_archives, without its fixed pair of file names."""
    require(len(expected_columns) == len(set(expected_columns)) == 21, "expected 21 ordered columns")
    sums, counts, occurrences, files = {}, {}, Counter(), []
    sentinel_count = 0
    for path in paths:
        row_count, dates = 0, []
        with zipfile.ZipFile(path) as bundle:
            names = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
            require(len(names) == 1, "each Jena archive must contain exactly one CSV")
            with bundle.open(names[0]) as binary:
                reader = csv.reader(line.decode("latin-1") for line in binary)
                header = next(reader)
                require(len(header) == 22 and header[1:] == list(expected_columns), "header differs from the fixed 2020 numeric order")
                for row in reader:
                    if not row:
                        continue
                    stamp = datetime.strptime(row[0], "%d.%m.%Y %H:%M:%S")
                    require(datetime(year, 1, 1) <= stamp <= datetime(year + 1, 1, 1), "date outside the archive year; do not substitute or filter")
                    values = np.asarray([float(value) for value in row[1:]], dtype=np.float64)
                    require(len(values) == 21 and np.isfinite(values).all(), "Jena rows must contain 21 finite numeric values")
                    valid = values != -9999.0
                    sentinel_count += int((~valid).sum())
                    sums[stamp] = sums.get(stamp, np.zeros_like(values)) + np.where(valid, values, 0.0)
                    counts[stamp] = counts.get(stamp, np.zeros_like(values, dtype=np.int64)) + valid
                    occurrences[stamp] += 1
                    row_count += 1
                    dates.append(stamp)
        require(bool(dates), "empty archive")
        files.append({"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size, "raw_rows": row_count,
                      "first_timestamp": min(dates).isoformat(), "last_timestamp": max(dates).isoformat()})
    first, last = min(sums), max(sums)
    require(not first.second and not last.second, "Jena endpoints must align to whole minutes")
    size = int((last - first) / STEP) + 1
    values_np = np.zeros((size, 21), dtype=np.float32)
    mask_np = np.zeros_like(values_np, dtype=bool)
    present = 0
    for index in range(size):
        stamp = first + index * STEP
        if stamp in sums:
            valid = counts[stamp] > 0
            values_np[index, valid] = (sums[stamp][valid] / counts[stamp][valid]).astype(np.float32)
            mask_np[index, valid] = True
            present += 1
    metadata = {
        "record_id": f"weather:temporal{year}", "archives": files, "numeric_columns": list(expected_columns),
        "first_timestamp": first.isoformat(), "last_timestamp": last.isoformat(), "lattice_rows": size,
        "raw_rows": sum(occurrences.values()), "unique_timestamps": len(sums),
        "duplicate_extra_rows": sum(n - 1 for n in occurrences.values()),
        "sentinel_entries_before_averaging": sentinel_count, "observed_per_channel": mask_np.sum(0).tolist(),
        "missing_entries_after_lattice": int((~mask_np).sum()), "absent_timestamp_rows": size - present,
        "off_lattice_unique_timestamps": len(sums) - present,
        "semantics": "Latin-1; strict datetime; float64 valid duplicate sums/counts; float32 mean; -9999 missing; complete min/max ten-minute lattice",
    }
    return (f"weather:temporal{year}", torch.arange(size, dtype=torch.float32), torch.from_numpy(values_np), torch.from_numpy(mask_np)), metadata


def scaled_record(raw, scaler_path: Path, record_id: str):
    """Apply the unchanged 2020 training scaler; it is never refitted."""
    require(sha256(scaler_path) == WEATHER_SCALER_SHA256, "original 2020 training scaler hash mismatch")
    values, mask = load_scaler(scaler_path, 21).transform(raw[2], raw[3])
    require(bool(torch.isfinite(values).all()), "nonfinite transformed values")
    return isolated_record((record_id, raw[1], values, mask))


def roster_report(payload, metadata) -> dict:
    bundles = {arm: build_roster(payload, metadata, arm) for arm in CONFIG["coordinate_arms"]}
    paired_across_arms(bundles)
    proof = bundles["LL"].proof
    reasons = dict(sorted(Counter(block["reason"] for block in proof["excluded_blocks"]).items()))
    return {key: proof[key] for key in proof if key != "excluded_blocks"} | {"excluded_block_reasons": reasons, "windows": len(bundles["LL"].dev_score)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-2024", type=Path, required=True, help=f"official {ARCHIVE}")
    parser.add_argument("--weather-2020a", type=Path, required=True, help="pinned mpi_roof_2020a.zip, read for its header line only")
    parser.add_argument("--original-data", type=Path, required=True, help="directory holding the original weather/scaler.npz")
    parser.add_argument("--out", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    require_protocol()
    require(np.__version__ == CONFIG["evaluation"]["numpy_version"], "study preparation requires NumPy 1.26.4")
    require(args.weather_2024.name == ARCHIVE, f"exactly the official archive {ARCHIVE} is allowed")
    torch.set_num_threads(1)
    verify_source(args.weather_2020a, "weather_2020a")
    expected = header_of(args.weather_2020a)[1:]
    require(header_of(args.weather_2024)[1:] == expected, f"{YEAR} numeric headers differ from the 2020 archives (precondition b)")
    raw, metadata = read_archives([args.weather_2024], expected, YEAR)
    payload = scaled_record(raw, args.original_data / "weather" / "scaler.npz", RECORD_ID)
    metadata.update(scaler_sha256=WEATHER_SCALER_SHA256, scaler_refitted=False, header_matches_2020=True)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(payload, args.out / PAYLOAD_FILE)
    write_json(args.out / METADATA_FILE, metadata)
    report = roster_report(payload, metadata)
    write_json(args.out / "PREPARATION.json", {
        "protocol_sha256": PROTOCOL_SHA256,
        "scope": "confirmation payload; values were not printed, plotted or summarised",
        ROSTER: {key: metadata[key] for key in ("archives", "first_timestamp", "last_timestamp", "lattice_rows", "raw_rows",
                                                "duplicate_extra_rows", "absent_timestamp_rows", "header_matches_2020", "scaler_refitted")},
        "rosters": {ROSTER: report},
        "files": {name: sha256(args.out / name) for name in (PAYLOAD_FILE, METADATA_FILE)},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    })
    print("Prepared. Realized roster counts (counts only):")
    print(f"  {ROSTER}: anchors {report['realized_anchor_count']} of {report['qualifying_anchor_count']} qualifying, "
          f"source blocks {report['source_block_count']['score']}, excluded blocks {report['excluded_block_count']}, windows {report['windows']}")
    print(f"  timestamps {metadata['first_timestamp']} to {metadata['last_timestamp']}, lattice rows {metadata['lattice_rows']}, "
          f"absent rows {metadata['absent_timestamp_rows']}")


if __name__ == "__main__":
    main()
