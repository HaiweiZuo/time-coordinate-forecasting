#!/usr/bin/env python
"""Prepare the three confirmation payloads (protocol section 4). CPU only.

The script never prints, plots or summarises observation values of the
confirmation records. It reports identities, counts and hashes only.

* USHCN test role: the records outside the training (668) and validation
  (222) roles under the original entity permutation, transformed with the
  original training scaler.
* Weather 2020 test role: the lattice rows after the 70% training and 10%
  validation slices, transformed with the original 2020 training scaler.
* Weather 2023: the two official roof-station archives, loaded with the
  semantics of the frozen 2022 temporal protocol and the same 2020 scaler.

The regenerated training and validation payloads are compared with the
original study payloads first, so the same code path is known to reproduce
the development inputs before it is applied to unseen records.
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

import numpy as np
import torch

from confirmation.common import (
    CONFIG, PROTOCOL_SHA256, ROOT, TEMPORAL_RECORD_ID, TEMPORAL_YEAR, WEATHER_SCALER_SHA256,
    WEATHER_TEST_RECORD_ID, read_json, require, require_protocol, sha256, write_json,
)
from confirmation.rosters import build_roster, paired_across_arms
from prepare_inputs import isolated_record, load_pinned_ushcn, verify_source
from preprocessing import ENTITY_SPLIT_SEED, Scaler, _record, load_jena_weather, prepare_ushcn, prepare_weather

STEP = timedelta(minutes=10)
ARCHIVES = (f"mpi_roof_{TEMPORAL_YEAR}a.zip", f"mpi_roof_{TEMPORAL_YEAR}b.zip")


def load_scaler(path: Path, channels: int) -> Scaler:
    with np.load(path, allow_pickle=False) as stored:
        require(set(stored.files) == {"mean", "scale", "count"}, "unexpected scaler fields")
        scaler = Scaler(stored["mean"].copy(), stored["scale"].copy(), stored["count"].copy())
    require(scaler.mean.shape == (channels,) and np.all(scaler.count > 0) and np.all(scaler.scale > 0), "invalid original scaler")
    return scaler


def same_scaler(left: Scaler, right: Scaler) -> dict:
    return {
        "mean_max_abs_difference": float(np.max(np.abs(left.mean - right.mean))),
        "scale_max_abs_difference": float(np.max(np.abs(left.scale - right.scale))),
        "count_equal": bool(np.array_equal(left.count, right.count)),
    }


def compare_records(regenerated, original) -> dict:
    """Development payloads only: identity and largest tensor difference."""
    require(len(regenerated) == len(original), "regenerated development payload has a different record count")
    exact, largest = True, 0.0
    for new, old in zip(regenerated, original):
        require(str(new[0]) == str(old[0]), "regenerated development record order differs")
        for a, b in zip(new[1:], old[1:]):
            a, b = torch.as_tensor(a), torch.as_tensor(b)
            require(a.shape == b.shape, "regenerated development tensor shape differs")
            if not torch.equal(a.to(b.dtype), b):
                exact = False
                largest = max(largest, float((a.double() - b.double()).abs().max()))
    return {"records": len(original), "exact_match": exact, "max_abs_difference": largest}


def prepare_ushcn_test(source: Path, original: Path) -> tuple[list, dict]:
    records = [_record(record) for record in load_pinned_ushcn(source)]
    by_id = {str(record[0]): record for record in records}
    require(len(by_id) == len(records), "record_id values must be unique")
    ids = sorted(by_id)
    order = np.random.default_rng(ENTITY_SPLIT_SEED).permutation(len(ids))
    shuffled = [ids[index] for index in order]
    n_train, n_val = int(0.6 * len(ids)), int(0.2 * len(ids))
    test_ids = shuffled[n_train + n_val:]
    regenerated, portable_scaler = prepare_ushcn(records)
    require([str(r[0]) for r in regenerated["train"]] == shuffled[:n_train], "training role differs from the original permutation")
    require([str(r[0]) for r in regenerated["val"]] == shuffled[n_train:n_train + n_val], "validation role differs from the original permutation")
    original_scaler = load_scaler(original / "ushcn" / "scaler.npz", 5)
    checks = {
        "scaler": same_scaler(portable_scaler, original_scaler),
        "train": compare_records(regenerated["train"], torch.load(original / "ushcn" / "train.pt", map_location="cpu", weights_only=True)),
        "val": compare_records(regenerated["val"], torch.load(original / "ushcn" / "val.pt", map_location="cpu", weights_only=True)),
    }
    for split in ("train", "val"):
        require(checks[split]["max_abs_difference"] <= 1e-6, f"regenerated USHCN {split} payload differs from the original study payload")
    payload = []
    for record_id in test_ids:
        source_record = by_id[record_id]
        values, mask = original_scaler.transform(source_record[2], source_record[3])
        payload.append(isolated_record((source_record[0], source_record[1], values, mask)))
    used = set(shuffled[:n_train + n_val])
    require(not used & set(test_ids) and len(test_ids) == len(ids) - n_train - n_val, "test role overlaps a development role")
    return payload, {
        "source_records": len(ids), "train_records": n_train, "val_records": n_val, "test_records": len(test_ids),
        "entity_split_seed": ENTITY_SPLIT_SEED, "scaler_sha256": sha256(original / "ushcn" / "scaler.npz"),
        "scaler_refitted": False, "development_regeneration_checks": checks,
    }


def prepare_weather_test(archives: list[Path], original: Path) -> tuple[tuple, dict]:
    verify_source(archives[0], "weather_2020a")
    verify_source(archives[1], "weather_2020b")
    record = load_jena_weather(archives)
    record_id, times, values, mask = _record(record)
    train_end = int(len(times) * 0.7)
    val_end = train_end + int(len(times) * 0.1)
    require(0 < train_end < val_end < len(times), "series is too short for the original partition fractions")
    regenerated, portable_scaler = prepare_weather(record)
    scaler_path = original / "weather" / "scaler.npz"
    require(sha256(scaler_path) == WEATHER_SCALER_SHA256, "original 2020 training scaler hash mismatch")
    original_scaler = load_scaler(scaler_path, 21)
    checks = {
        "scaler": same_scaler(portable_scaler, original_scaler),
        "train": compare_records([regenerated["train"]], [torch.load(original / "weather" / "train.pt", map_location="cpu", weights_only=True)]),
        "val": compare_records([regenerated["val"]], [torch.load(original / "weather" / "val.pt", map_location="cpu", weights_only=True)]),
    }
    for split in ("train", "val"):
        require(checks[split]["max_abs_difference"] <= 1e-6, f"regenerated Weather {split} payload differs from the original study payload")
    scaled, finite_mask = original_scaler.transform(values[val_end:], mask[val_end:])
    payload = isolated_record((WEATHER_TEST_RECORD_ID, times[val_end:], scaled, finite_mask))
    return payload, {
        "lattice_rows": int(len(times)), "train_rows": train_end, "val_rows": val_end - train_end,
        "test_rows": int(len(times) - val_end), "record_id": WEATHER_TEST_RECORD_ID,
        "scaler_sha256": WEATHER_SCALER_SHA256, "scaler_refitted": False, "development_regeneration_checks": checks,
    }


def header_of(archive: Path) -> list[str]:
    """One metadata line only; no observation row is parsed."""
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
        require(len(names) == 1, "each Jena archive must contain exactly one CSV")
        with bundle.open(names[0]) as stream:
            return next(csv.reader([stream.readline().decode("latin-1")]))


def read_year_archives(paths: list[Path], expected_columns: list[str], year: int):
    """2022 temporal semantics: valid-duplicate mean, -9999 missing, complete ten-minute lattice."""
    require(tuple(path.name for path in paths) == (f"mpi_roof_{year}a.zip", f"mpi_roof_{year}b.zip"),
            "exactly the two ordered archives of the fixed year are allowed")
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


def prepare_weather_temporal(archives: list[Path], reference_archive: Path, original: Path, year: int = TEMPORAL_YEAR) -> tuple[tuple, dict]:
    expected = header_of(reference_archive)[1:]
    headers = [header_of(path)[1:] for path in archives]
    require(all(header == expected for header in headers), f"{year} numeric headers differ from the 2020 archives (precondition b)")
    payload, metadata = read_year_archives(archives, expected, year)
    scaler_path = original / "weather" / "scaler.npz"
    require(sha256(scaler_path) == WEATHER_SCALER_SHA256, "original 2020 training scaler hash mismatch")
    scaler = load_scaler(scaler_path, 21)
    values, mask = scaler.transform(payload[2], payload[3])
    require(bool(torch.isfinite(values).all()), "nonfinite transformed values")
    metadata.update(scaler_sha256=WEATHER_SCALER_SHA256, scaler_refitted=False, header_matches_2020=True)
    return isolated_record((f"weather:temporal{year}", payload[1], values, mask)), metadata


def roster_report(name: str, payload, metadata=None) -> dict:
    bundles = {arm: build_roster(name, payload, arm, metadata) for arm in CONFIG["coordinate_arms"]}
    paired_across_arms(bundles)
    proof = bundles["LL"].proof
    return {key: proof[key] for key in proof if key != "excluded_blocks"} | {"windows": len(bundles["LL"].dev_score)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ushcn-source", type=Path, required=True, help="pinned processed ushcn.pt")
    parser.add_argument("--weather-2020", type=Path, nargs=2, required=True, metavar=("A", "B"))
    parser.add_argument("--weather-temporal", type=Path, nargs=2, required=True, metavar=("A", "B"))
    parser.add_argument("--original-data", type=Path, required=True, help="directory holding the original weather/ and ushcn/ train.pt, val.pt, scaler.npz")
    parser.add_argument("--out", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    require_protocol()
    require(np.__version__ == CONFIG["evaluation"]["numpy_version"], "study preparation requires NumPy 1.26.4")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    ushcn, ushcn_meta = prepare_ushcn_test(args.ushcn_source, args.original_data)
    weather_test, weather_test_meta = prepare_weather_test(list(args.weather_2020), args.original_data)
    temporal, temporal_meta = prepare_weather_temporal(list(args.weather_temporal), args.weather_2020[0], args.original_data)
    torch.save(ushcn, args.out / "ushcn_test.pt")
    torch.save(weather_test, args.out / "weather2020_test.pt")
    torch.save(temporal, args.out / "weather2023.pt")
    write_json(args.out / "weather2023_metadata.json", temporal_meta)
    rosters = {
        "ushcn_test": roster_report("ushcn_test", ushcn),
        "weather2020_test": roster_report("weather2020_test", weather_test),
        "weather2023": roster_report("weather2023", temporal, temporal_meta),
    }
    receipt = {
        "protocol_sha256": PROTOCOL_SHA256,
        "scope": "confirmation payloads; values were not printed, plotted or summarised",
        "ushcn_test": ushcn_meta, "weather2020_test": weather_test_meta,
        "weather2023": {key: temporal_meta[key] for key in ("archives", "first_timestamp", "last_timestamp", "lattice_rows",
                                                             "absent_timestamp_rows", "header_matches_2020", "scaler_refitted")},
        "rosters": rosters,
        "files": {name: sha256(args.out / name) for name in ("ushcn_test.pt", "weather2020_test.pt", "weather2023.pt", "weather2023_metadata.json")},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    }
    write_json(args.out / "PREPARATION.json", receipt)
    print("Prepared. Realized roster counts (counts only):")
    for name, report in rosters.items():
        print(f"  {name}: anchors {report['realized_anchor_count']} of {report['qualifying_anchor_count']} qualifying, "
              f"source blocks {report['source_block_count']['score']}, windows {report['windows']}")


if __name__ == "__main__":
    main()
