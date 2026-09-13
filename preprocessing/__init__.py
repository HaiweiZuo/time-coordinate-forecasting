"""Portable excerpts of the study's observed-training-only preprocessing.

Only training and validation payloads are produced. Record identifiers and
absolute source-index times retain the original split semantics. See
source_spec.json for upstream inputs and original source-file hashes.
"""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

Record = tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]
ENTITY_SPLIT_SEED = 20260801


def _record(record: Sequence[object]) -> Record:
    if len(record) != 4:
        raise ValueError("each record must be (record_id, times, values, mask)")
    record_id, times, values, mask = record
    times = torch.as_tensor(times, dtype=torch.float32).cpu()
    values = torch.as_tensor(values, dtype=torch.float32).cpu()
    mask = torch.as_tensor(mask, dtype=torch.bool).cpu()
    if times.ndim != 1 or values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("expected times [T] and matching values/mask [T,D]")
    if len(times) != len(values) or len(times) == 0:
        raise ValueError("record must contain at least one aligned time point")
    if not torch.isfinite(times).all() or torch.any(times[1:] < times[:-1]):
        raise ValueError("times must be finite and sorted")
    mask = mask & torch.isfinite(values)
    values = torch.where(mask, values, torch.zeros_like(values))
    return record_id, times, values, mask


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    scale: np.ndarray
    count: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez(Path(path), mean=self.mean, scale=self.scale, count=self.count)

    def transform(self, values: object, mask: object) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.as_tensor(values, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.bool) & torch.isfinite(values)
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        scale = torch.as_tensor(self.scale, dtype=values.dtype, device=values.device)
        if values.ndim != 2 or values.shape != mask.shape or values.shape[1] != len(mean):
            raise ValueError("values/mask shape does not match scaler")
        normalized = torch.where(mask, (values - mean) / scale, torch.zeros_like(values))
        return normalized, mask


def fit_scaler(train_records: Iterable[Record]) -> Scaler:
    """Original float64 pooled population mean/std, in original record order."""
    mean = m2 = count = None
    for record in train_records:
        _, _, values_t, mask_t = _record(record)
        values = values_t.numpy().astype(np.float64, copy=False)
        observed = mask_t.numpy() & np.isfinite(values)
        if mean is None:
            mean = np.zeros(values.shape[1], dtype=np.float64)
            m2 = np.zeros_like(mean)
            count = np.zeros(values.shape[1], dtype=np.int64)
        elif values.shape[1] != len(mean):
            raise ValueError("all records must have the same feature count")
        for channel in range(values.shape[1]):
            batch = values[observed[:, channel], channel]
            if not len(batch):
                continue
            batch_count = len(batch)
            batch_mean = batch.mean()
            batch_m2 = np.square(batch - batch_mean).sum()
            new_count = count[channel] + batch_count
            delta = batch_mean - mean[channel]
            mean[channel] += delta * batch_count / new_count
            m2[channel] += batch_m2 + delta * delta * count[channel] * batch_count / new_count
            count[channel] = new_count
    if count is None or np.any(count == 0):
        raise ValueError("every feature needs a finite observed training value")
    scale = np.sqrt(m2 / count)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    return Scaler(mean, scale, count)


def load_jena_weather(paths: Sequence[str | Path], record_id: str = "weather") -> Record:
    """Original loader: Latin-1, valid duplicate means and a ten-minute lattice."""
    if not paths:
        raise ValueError("at least one Jena archive is required")
    columns: list[str] | None = None
    sums: dict[datetime, np.ndarray] = {}
    counts: dict[datetime, np.ndarray] = {}
    for path_like in paths:
        path = Path(path_like)
        with zipfile.ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError("each Jena archive must contain exactly one CSV")
            with archive.open(csv_names[0]) as binary:
                lines = (line.decode("latin-1") for line in binary)
                reader = csv.reader(lines)
                header = next(reader)
                numeric_columns = header[1:]
                if columns is None:
                    columns = numeric_columns
                elif columns != numeric_columns:
                    raise ValueError("Jena archive columns changed")
                for row in reader:
                    if not row:
                        continue
                    timestamp = datetime.strptime(row[0], "%d.%m.%Y %H:%M:%S")
                    values = np.asarray([float(value) for value in row[1:]], dtype=np.float64)
                    if len(values) != len(columns) or not np.isfinite(values).all():
                        raise ValueError("Jena rows must contain finite numeric values")
                    valid = values != -9999.0
                    sums[timestamp] = sums.get(timestamp, np.zeros_like(values)) + np.where(valid, values, 0.0)
                    counts[timestamp] = counts.get(timestamp, np.zeros_like(values, dtype=np.int64)) + valid
    if not sums:
        raise ValueError("Jena archives contain no observations")
    first, last = min(sums), max(sums)
    if first.second or last.second:
        raise ValueError("Jena timestamps must align to whole minutes")
    steps = int((last - first) / timedelta(minutes=10)) + 1
    values_np = np.zeros((steps, len(columns)), dtype=np.float32)
    mask_np = np.zeros_like(values_np, dtype=bool)
    for index in range(steps):
        timestamp = first + index * timedelta(minutes=10)
        if timestamp in sums:
            valid = counts[timestamp] > 0
            values_np[index, valid] = (sums[timestamp][valid] / counts[timestamp][valid]).astype(np.float32)
            mask_np[index, valid] = True
    values, mask = torch.from_numpy(values_np), torch.from_numpy(mask_np)
    return str(record_id), torch.arange(steps, dtype=torch.float32), values, mask


def prepare_weather(record: Record) -> tuple[dict[str, Record], Scaler]:
    """Keep the original 70%/10% source-row slices; do not construct the remainder."""
    record_id, times, values, mask = _record(record)
    train_end = int(len(times) * 0.7)
    val_end = train_end + int(len(times) * 0.1)
    if not 0 < train_end < val_end < len(times):
        raise ValueError("series is too short for the original partition fractions")
    splits = {
        split: (f"{record_id}:{split}", times[left:right], values[left:right], mask[left:right])
        for split, left, right in (("train", 0, train_end), ("val", train_end, val_end))
    }
    scaler = fit_scaler([splits["train"]])
    transformed = {}
    for split, value in splits.items():
        rid, time, val, observed = _record(value)
        normalized, finite_mask = scaler.transform(val, observed)
        transformed[split] = (rid, time, normalized, finite_mask)
    return transformed, scaler


def prepare_ushcn(records: Sequence[Record]) -> tuple[dict[str, list[Record]], Scaler]:
    """Original ID-order permutation and 60%/20% selection, without remainder output."""
    records = [_record(record) for record in records]
    by_id = {str(record[0]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("record_id values must be unique within a source")
    ids = sorted(by_id)
    order = np.random.default_rng(ENTITY_SPLIT_SEED).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    n_train, n_val = int(0.6 * len(ids)), int(0.2 * len(ids))
    if not n_train or not n_val or n_train + n_val >= len(ids):
        raise ValueError("source is too small for the original partition fractions")
    selections = {"train": shuffled[:n_train], "val": shuffled[n_train:n_train + n_val]}
    scaler = fit_scaler(by_id[record_id] for record_id in selections["train"])
    scaled = {}
    for split, selected_ids in selections.items():
        scaled[split] = []
        for record_id in selected_ids:
            source_record = by_id[record_id]
            values, mask = scaler.transform(source_record[2], source_record[3])
            scaled[split].append((source_record[0], source_record[1], values, mask))
    return scaled, scaler
