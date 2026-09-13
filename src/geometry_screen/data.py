"""Strict payload validation and fixed-shape, leakage-safe development windows."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


MASK_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
}


def require_development_split(split: str) -> str:
    if split not in {"train", "val"}:
        raise ValueError("only development train/val splits are permitted; test is terminally forbidden")
    return split


@dataclass(frozen=True)
class GeometrySpec:
    name: str
    physical_step: int
    gap: int
    history_points: int
    horizon_points: int
    stride: int
    t_ref: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "GeometrySpec":
        required = {
            "name", "physical_step", "gap", "history_points", "horizon_points", "stride", "t_ref"
        }
        if set(value) != required:
            raise ValueError(f"stratum keys must be exactly {sorted(required)}")
        spec = cls(
            str(value["name"]),
            int(value["physical_step"]),
            int(value["gap"]),
            int(value["history_points"]),
            int(value["horizon_points"]),
            int(value["stride"]),
            int(value["t_ref"]),
        )
        if spec.physical_step not in {1, 2}:
            raise ValueError("physical_step must be 1 or 2")
        if min(spec.history_points, spec.horizon_points, spec.stride, spec.t_ref) <= 0 or spec.gap < 0:
            raise ValueError("invalid stratum geometry")
        if spec.span > spec.t_ref:
            raise ValueError("stratum span exceeds fixed T_ref")
        return spec

    @property
    def span(self) -> int:
        return (
            self.history_points * self.physical_step
            + self.gap
            + self.horizon_points * self.physical_step
        )


@dataclass(frozen=True)
class StrictRecord:
    record_id: str
    times: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class PartitionBundle:
    fit: "WindowDataset"
    early_stop: "WindowDataset"
    dev_score: "WindowDataset"
    proof: dict[str, object]
    channels: int


class WindowDataset(Dataset):
    def __init__(self, samples: Sequence[Mapping[str, object]], expected_strata: Sequence[str]) -> None:
        self.samples = list(samples)
        self.expected_strata = tuple(expected_strata)
        counts = {name: 0 for name in self.expected_strata}
        for sample in self.samples:
            name = str(sample["stratum"])
            if name not in counts:
                raise ValueError(f"unregistered stratum {name!r}")
            counts[name] += 1
        if not self.samples or any(count == 0 for count in counts.values()):
            raise ValueError(f"every registered stratum must be nonempty: {counts}")
        self.counts = counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        return self.samples[index]


def _digest(*parts: object) -> str:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_mask(value: object, shape: torch.Size) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype not in MASK_DTYPES:
        raise ValueError("mask must be a supported CPU tensor")
    if value.shape != shape:
        raise ValueError("mask/value shapes differ")
    if value.dtype != torch.bool and not torch.isfinite(value).all():
        raise ValueError("mask must be finite before conversion")
    if not torch.all((value == 0) | (value == 1)):
        raise ValueError("mask must contain only original values 0 or 1")
    return value.to(dtype=torch.bool)


def _strict_record(value: object) -> StrictRecord:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("record must be exactly (record_id,times,values,mask)")
    record_id, times, values, raw_mask = value
    if isinstance(record_id, bool) or not isinstance(record_id, (str, int)) or not str(record_id):
        raise ValueError("record_id must be a nonempty string or integer")
    record_id = str(record_id)
    if not isinstance(times, torch.Tensor) or times.device.type != "cpu" or times.dtype != torch.float32:
        raise ValueError("times must be a CPU float32 tensor")
    if not isinstance(values, torch.Tensor) or values.device.type != "cpu" or values.dtype != torch.float32:
        raise ValueError("values must be a CPU float32 tensor")
    if times.ndim != 1 or values.ndim != 2 or len(times) != len(values) or len(times) < 2:
        raise ValueError("times [T] and values [T,C] must be aligned and nonempty")
    if values.shape[1] < 1 or not torch.isfinite(times).all() or not torch.isfinite(values).all():
        raise ValueError("times/values must be finite with at least one channel")
    if not torch.all(times[1:] > times[:-1]):
        raise ValueError("times must be strictly increasing")
    mask = _strict_mask(raw_mask, values.shape)
    return StrictRecord(record_id, times.clone(), values.clone(), mask.clone())


def validate_payload(payload: object, dataset: str) -> list[StrictRecord]:
    if dataset == "weather_jena2020":
        records = [_strict_record(payload)]
    elif dataset == "ushcn":
        if not isinstance(payload, (tuple, list)) or not payload:
            raise ValueError("USHCN payload must be a nonempty record sequence")
        records = [_strict_record(record) for record in payload]
    else:
        raise ValueError(f"unsupported dataset {dataset!r}")
    identifiers = [record.record_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("record identifiers must be unique")
    channels = {record.values.shape[1] for record in records}
    if len(channels) != 1:
        raise ValueError("channels must be consistent within a payload")
    return records


def encode_time_labels(spec: GeometrySpec, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "legacy":
        return (
            torch.arange(spec.history_points, dtype=torch.float32) / spec.history_points,
            torch.arange(spec.horizon_points, dtype=torch.float32) / spec.horizon_points,
        )
    if mode != "geometry":
        raise ValueError("time encoding must be legacy or geometry")
    history = torch.arange(spec.history_points, dtype=torch.float32) * spec.physical_step
    query = (
        spec.history_points * spec.physical_step
        + spec.gap
        + torch.arange(spec.horizon_points, dtype=torch.float32) * spec.physical_step
    )
    return history / spec.t_ref, query / spec.t_ref


def _weather_lattice(record: StrictRecord) -> tuple[int, torch.Tensor, torch.Tensor]:
    rounded = record.times.round()
    if not torch.allclose(record.times, rounded, atol=1e-5, rtol=0.0):
        raise ValueError("Weather times must lie on the frozen integer lattice")
    indices = rounded.to(torch.int64)
    origin = int(indices[0])
    relative = indices - origin
    if not torch.equal(relative, torch.arange(len(relative), dtype=torch.int64)):
        raise ValueError("Weather payload must retain its complete 10-minute lattice")
    return origin, record.values, record.mask


def _weather_point_delete(sample: dict[str, object], rate: float, mask_seed: int) -> None:
    if not 0 <= rate < 1:
        raise ValueError("Weather missing rate must be in [0,1)")
    observed = torch.as_tensor(sample["x_mask"], dtype=torch.bool)
    positions = observed.nonzero(as_tuple=False).tolist()
    hide = min(int(round(len(positions) * rate)), max(len(positions) - 1, 0))
    ranked = sorted(
        positions,
        key=lambda position: hashlib.sha256(
            f"{mask_seed}|{sample['anchor_digest']}|{position[0]}|{position[1]}".encode("utf-8")
        ).digest(),
    )
    for row, channel in ranked[:hide]:
        observed[row, channel] = False
    if not observed.any():
        raise ValueError("Weather point deletion removed all context")
    sample["x_mask"] = observed
    sample["x"] = torch.where(observed, torch.as_tensor(sample["x"]), 0.0)


def _sample_at_regular(
    values: torch.Tensor,
    mask: torch.Tensor,
    origin: int,
    start: int,
    record_id: str,
    spec: GeometrySpec,
    mode: str,
    block_span: int,
) -> dict[str, object] | None:
    x_positions = start + torch.arange(spec.history_points) * spec.physical_step
    query_start = start + spec.history_points * spec.physical_step + spec.gap
    y_positions = query_start + torch.arange(spec.horizon_points) * spec.physical_step
    if int(y_positions[-1]) >= len(values):
        return None
    x_mask, y_mask = mask[x_positions], mask[y_positions]
    if not x_mask.any() or not y_mask.any():
        return None
    physical_start = origin + start
    physical_end = physical_start + spec.span
    block_index = math.floor(physical_start / block_span)
    block_start = block_index * block_span
    if physical_end > block_start + block_span:
        return None
    anchor_digest = _digest("anchor", record_id, physical_start)
    block_digest = _digest("source_block", record_id, block_index)
    x_time, y_time = encode_time_labels(spec, mode)
    return {
        "sample_id": _digest("sample", anchor_digest, spec.name),
        "anchor_digest": anchor_digest,
        "source_block_digest": block_digest,
        "stratum": spec.name,
        "physical_step": spec.physical_step,
        "gap": spec.gap,
        "x": torch.where(x_mask, values[x_positions], 0.0),
        "x_mark": x_time,
        "x_mask": x_mask.clone(),
        "y": torch.where(y_mask, values[y_positions], 0.0),
        "y_mark": y_time,
        "y_mask": y_mask.clone(),
        "source_bounds": (
            physical_start,
            physical_start + spec.history_points * spec.physical_step,
            physical_start + spec.history_points * spec.physical_step + spec.gap,
            physical_end,
        ),
    }


def _rank_select(indices: torch.Tensor, count: int) -> torch.Tensor | None:
    if indices.numel() < count:
        return None
    ranks = torch.linspace(0, indices.numel() - 1, count, dtype=torch.float64).round().to(torch.int64)
    selected = indices[ranks]
    if len(torch.unique(selected)) != count:
        raise ValueError("rank selection failed to produce a fixed unique shape")
    return selected


def _sample_at_irregular(
    record: StrictRecord,
    anchor: float,
    spec: GeometrySpec,
    mode: str,
    block_span: int,
) -> dict[str, object] | None:
    history_end = anchor + spec.history_points * spec.physical_step
    query_start = history_end + spec.gap
    physical_end = query_start + spec.horizon_points * spec.physical_step
    block_index = math.floor(anchor / block_span)
    if physical_end > (block_index + 1) * block_span + 1e-8:
        return None
    x_candidates = ((record.times >= anchor) & (record.times < history_end)).nonzero(as_tuple=False).flatten()
    y_candidates = ((record.times >= query_start) & (record.times < physical_end)).nonzero(as_tuple=False).flatten()
    x_positions = _rank_select(x_candidates, spec.history_points)
    y_positions = _rank_select(y_candidates, spec.horizon_points)
    if x_positions is None or y_positions is None:
        return None
    x_mask, y_mask = record.mask[x_positions], record.mask[y_positions]
    if not x_mask.any() or not y_mask.any():
        return None
    if mode == "legacy":
        x_time = (record.times[x_positions] - anchor) / (spec.history_points * spec.physical_step)
        y_time = (record.times[y_positions] - query_start) / (spec.horizon_points * spec.physical_step)
    elif mode == "geometry":
        x_time = (record.times[x_positions] - anchor) / spec.t_ref
        y_time = (record.times[y_positions] - anchor) / spec.t_ref
    else:
        raise ValueError("time encoding must be legacy or geometry")
    anchor_digest = _digest("anchor", record.record_id, float(anchor))
    block_digest = _digest("source_block", record.record_id, block_index)
    return {
        "sample_id": _digest("sample", anchor_digest, spec.name),
        "anchor_digest": anchor_digest,
        "source_block_digest": block_digest,
        "stratum": spec.name,
        "physical_step": spec.physical_step,
        "gap": spec.gap,
        "x": torch.where(x_mask, record.values[x_positions], 0.0),
        "x_mark": x_time.to(torch.float32),
        "x_mask": x_mask.clone(),
        "y": torch.where(y_mask, record.values[y_positions], 0.0),
        "y_mark": y_time.to(torch.float32),
        "y_mask": y_mask.clone(),
        "source_bounds": (float(anchor), float(history_end), float(query_start), float(physical_end)),
    }


def _validate_registered_strata(values: Sequence[Mapping[str, object]], dataset: str) -> list[GeometrySpec]:
    specs = [GeometrySpec.from_mapping(value) for value in values]
    if len(specs) != 4 or len({spec.name for spec in specs}) != 4:
        raise ValueError("exactly four uniquely named strata are required")
    expected_gaps = {0, 18} if dataset == "weather_jena2020" else {0, 6}
    if {(spec.physical_step, spec.gap) for spec in specs} != {
        (step, gap) for step in (1, 2) for gap in expected_gaps
    }:
        raise ValueError("strata must be the registered physical_step x gap 2x2")
    fixed = {(spec.history_points, spec.horizon_points, spec.stride, spec.t_ref) for spec in specs}
    expected = (72, 18, 36, 198) if dataset == "weather_jena2020" else (12, 6, 6, 42)
    if fixed != {expected}:
        raise ValueError("fixed tensor shape/stride/T_ref contract mismatch")
    return specs


def _anchor_groups(
    records: Sequence[StrictRecord],
    dataset: str,
    specs: Sequence[GeometrySpec],
    mode: str,
    block_span: int,
    missing_rate: float,
    mask_seed: int,
) -> dict[str, dict[str, dict[str, object]]]:
    groups: dict[str, dict[str, dict[str, object]]] = {}
    maximum_span = max(spec.span for spec in specs)
    stride = specs[0].stride
    for record in records:
        if dataset == "weather_jena2020":
            origin, values, mask = _weather_lattice(record)
            anchors: Sequence[tuple[float, int | None]] = [
                (float(origin + start), start) for start in range(0, len(values) - maximum_span + 1, stride)
            ]
        else:
            first = math.ceil(float(record.times[0]) / stride) * stride
            last = float(record.times[-1]) - maximum_span
            if last < first:
                continue
            anchors = [(float(anchor), None) for anchor in torch.arange(first, last + 1e-6, stride).tolist()]
        for physical_anchor, regular_start in anchors:
            candidates: dict[str, dict[str, object]] = {}
            for spec in specs:
                sample = (
                    _sample_at_regular(
                        values, mask, origin, int(regular_start), record.record_id, spec, mode, block_span
                    )
                    if dataset == "weather_jena2020"
                    else _sample_at_irregular(record, physical_anchor, spec, mode, block_span)
                )
                if sample is None:
                    candidates = {}
                    break
                candidates[spec.name] = sample
            if len(candidates) != len(specs):
                continue
            anchor = str(next(iter(candidates.values()))["anchor_digest"])
            if dataset == "weather_jena2020":
                for sample in candidates.values():
                    _weather_point_delete(sample, missing_rate, mask_seed)
            groups[anchor] = candidates
    return groups


def _fold(block_digest: str, salt: str) -> str:
    value = int(hashlib.sha256(f"{salt}|{block_digest}".encode("ascii")).hexdigest()[:16], 16)
    return "early_stop" if value % 5 == 0 else "fit"


def _select(
    groups: Mapping[str, Mapping[str, dict[str, object]]],
    cap: int,
    minimum: int,
) -> tuple[list[dict[str, object]], list[str], set[str]]:
    if cap < minimum or minimum < 1:
        raise ValueError("partition cap/minimum contract is invalid")
    roster = sorted(groups)[:cap]
    if len(roster) < minimum:
        raise ValueError(f"partition has {len(roster)} shared anchors, below registered minimum {minimum}")
    samples = [dict(groups[anchor][name]) for anchor in roster for name in sorted(groups[anchor])]
    blocks = {str(sample["source_block_digest"]) for sample in samples}
    return samples, roster, blocks


def _roster_digest(roster: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(roster)).encode("ascii")).hexdigest()


def build_partitions(
    train_payload: object,
    val_payload: object,
    dataset: str,
    dataset_config: Mapping[str, object],
    time_encoding: str,
    analysis_id: str,
) -> PartitionBundle:
    """Split train by source/time blocks; reserve frozen val for one dev score."""
    require_development_split("train")
    require_development_split("val")
    train_records = validate_payload(train_payload, dataset)
    val_records = validate_payload(val_payload, dataset)
    if {record.record_id for record in train_records} & {record.record_id for record in val_records}:
        raise ValueError("train and frozen validation source identifiers overlap")
    channels = train_records[0].values.shape[1]
    if any(record.values.shape[1] != channels for record in (*train_records, *val_records)):
        raise ValueError("channels must be consistent across train and validation payloads")
    specs = _validate_registered_strata(dataset_config["strata"], dataset)
    block_span = int(dataset_config["block_span"])
    if block_span != 4 * int(dataset_config["t_ref"]):
        raise ValueError("source/time block span must equal 4*T_ref")
    common = dict(
        dataset=dataset,
        specs=specs,
        mode=time_encoding,
        block_span=block_span,
        missing_rate=float(dataset_config["history_point_missing"]),
        mask_seed=int(dataset_config["mask_seed"]),
    )
    train_groups = _anchor_groups(train_records, **common)
    fit_groups: dict[str, Mapping[str, dict[str, object]]] = {}
    early_groups: dict[str, Mapping[str, dict[str, object]]] = {}
    for anchor, candidates in train_groups.items():
        block = str(next(iter(candidates.values()))["source_block_digest"])
        (early_groups if _fold(block, analysis_id) == "early_stop" else fit_groups)[anchor] = candidates
    dev_groups = _anchor_groups(val_records, **common)
    policy = dataset_config["partitions"]
    selected: dict[str, tuple[list[dict[str, object]], list[str], set[str]]] = {}
    for name, groups in (("fit", fit_groups), ("early_stop", early_groups), ("dev_score", dev_groups)):
        rules = policy[name]
        selected[name] = _select(groups, int(rules["cap_per_stratum"]), int(rules["minimum_per_stratum"]))
    names = [spec.name for spec in specs]
    fit = WindowDataset(selected["fit"][0], names)
    early = WindowDataset(selected["early_stop"][0], names)
    dev = WindowDataset(selected["dev_score"][0], names)
    fit_blocks, early_blocks = selected["fit"][2], selected["early_stop"][2]
    overlap = fit_blocks & early_blocks
    train_blocks = fit_blocks | early_blocks
    train_dev_overlap = train_blocks & selected["dev_score"][2]
    if overlap:
        raise ValueError("fit and early-stop source/time blocks overlap")
    if train_dev_overlap:
        raise ValueError("train and dev-score source/time blocks overlap")
    proof = {
        "schema_version": 5,
        "shared_anchor_selection": True,
        "fixed_tensor_shapes": {
            "history_points": specs[0].history_points,
            "horizon_points": specs[0].horizon_points,
        },
        "partition_counts": {
            "fit": fit.counts,
            "early_stop": early.counts,
            "dev_score": dev.counts,
        },
        "anonymous_roster_digest": {
            name: _roster_digest(selected[name][1]) for name in ("fit", "early_stop", "dev_score")
        },
        "source_block_digest": {
            name: _roster_digest(sorted(selected[name][2])) for name in ("fit", "early_stop", "dev_score")
        },
        "source_block_count": {
            name: len(selected[name][2]) for name in ("fit", "early_stop", "dev_score")
        },
        "fit_early_source_block_overlap": len(overlap),
        "fit_early_anchor_overlap": len(set(selected["fit"][1]) & set(selected["early_stop"][1])),
        "train_dev_source_id_overlap": 0,
        "train_dev_source_block_overlap": len(train_dev_overlap),
        "dev_score_source": "frozen_parent_val_payload_once",
        "dev_score_used_for_selection": False,
    }
    if proof["fit_early_anchor_overlap"] != 0:
        raise ValueError("fit and early-stop anchors overlap")
    return PartitionBundle(fit, early, dev, proof, int(channels))


def collate_windows(batch: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("empty batch")
    result: dict[str, object] = {
        "sample_id": [str(sample["sample_id"]) for sample in batch],
        "stratum": [str(sample["stratum"]) for sample in batch],
    }
    for key in ("x", "x_mark", "x_mask", "y", "y_mark", "y_mask"):
        result[key] = pad_sequence(
            [torch.as_tensor(sample[key]) for sample in batch], batch_first=True, padding_value=0
        )
    return result
