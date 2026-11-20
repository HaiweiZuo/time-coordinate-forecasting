"""Confirmation rosters built with the unchanged window rules of the study.

Three rosters (protocol section 4):

* ``ushcn_test`` and ``weather2020_test`` use the development roster rule:
  common anchor stride, all four strata or none, digest ordering, fixed cap,
  every qualifying anchor up to the cap, no minimum.
* ``weather2023`` uses the 2022 temporal rule: one anchor at the origin of
  every 792-index block, all four schedules or none, no cap, no replacement.

Only functions from src/geometry_screen/data.py build windows, masks and
digests. This file crosses the embedding (E) and raster (R) coordinates the
same way as training.make_partitions.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from types import SimpleNamespace

import torch

from confirmation.common import (
    CONFIG, ROSTER_CAPS, TEMPORAL_RECORD_ID, WEATHER_TEST_RECORD_ID, require,
)
from geometry_screen import data

STEP = timedelta(minutes=10)
PAIRED_IDENTITY = ("sample_id", "anchor_digest", "source_block_digest", "stratum", "source_bounds")
PAIRED_TENSORS = ("x", "y", "x_mask", "y_mask")


def _cross(embedding_samples, raster_samples):
    require(len(embedding_samples) == len(raster_samples), "unpaired coordinate roster")
    crossed = []
    for e, r in zip(embedding_samples, raster_samples):
        for key in PAIRED_IDENTITY:
            require(e[key] == r[key], "unpaired coordinate sample")
        for key in PAIRED_TENSORS:
            require(torch.equal(e[key], r[key]), "data or mask changed by coordinate factor")
        crossed.append(dict(e, x_mark=torch.stack((e["x_mark"], r["x_mark"]), dim=-1)))
    return crossed


def _capped(payload, dataset: str, cap: int, arm: str):
    settings = CONFIG["datasets"][dataset]
    records = data.validate_payload(payload, dataset)
    specs = data._validate_registered_strata(settings["strata"], dataset)
    block_span = int(settings["block_span"])
    require(block_span == 4 * int(settings["t_ref"]), "source/time block span must equal 4*T_ref")
    modes = CONFIG["coordinate_arms"][arm]
    built = {}
    for mode in sorted(set(modes.values())):
        groups = data._anchor_groups(
            records, dataset=dataset, specs=specs, mode=mode, block_span=block_span,
            missing_rate=float(settings["history_point_missing"]), mask_seed=int(settings["mask_seed"]),
        )
        samples, roster, blocks = data._select(groups, cap, 1)
        built[mode] = (samples, roster, blocks, len(groups))
    samples = _cross(built[modes["embedding"]][0], built[modes["raster"]][0])
    _, roster, blocks, qualifying = built[modes["embedding"]]
    require(built[modes["raster"]][1] == roster, "roster changed by coordinate factor")
    names = [spec.name for spec in specs]
    dataset_windows = data.WindowDataset(samples, names)
    proof = {
        "rule": "development roster rule: common stride, all four strata or none, digest order, cap, no minimum",
        "cap": cap,
        "source_records": len(records),
        "qualifying_anchor_count": qualifying,
        "realized_anchor_count": len(roster),
        "partition_counts": {"score": dataset_windows.counts},
        "anonymous_roster_digest": {"score": data._roster_digest(roster)},
        "source_block_digest": {"score": data._roster_digest(sorted(blocks))},
        "source_block_count": {"score": len(blocks)},
        "fixed_tensor_shapes": {"history_points": specs[0].history_points, "horizon_points": specs[0].horizon_points},
    }
    return SimpleNamespace(dev_score=dataset_windows, channels=int(records[0].values.shape[1]), proof=proof)


def _temporal(payload, metadata, arm: str):
    settings = CONFIG["datasets"]["weather_jena2020"]
    require((settings["block_span"], settings["t_ref"], settings["history_point_missing"], settings["mask_seed"])
            == (792, 198, 0.5, 99173), "frozen Weather settings changed")
    modes = CONFIG["coordinate_arms"][arm]
    record = data._strict_record(payload)
    require(record.record_id == TEMPORAL_RECORD_ID and record.values.shape[1] == 21, "temporal record identity/channel mismatch")
    origin, values, mask = data._weather_lattice(record)
    require(origin == 0 and len(values) == metadata["lattice_rows"], "temporal lattice must start at index zero")
    specs = data._validate_registered_strata(settings["strata"], "weather_jena2020")
    first = datetime.fromisoformat(metadata["first_timestamp"])
    maximum_span = max(spec.span for spec in specs)
    samples, blocks, excluded = [], [], []
    for start in range(0, len(values), 792):
        stamp = first + start * STEP
        block = {"block_index": start // 792, "anchor_index": start, "anchor_timestamp": stamp.isoformat(),
                 "calendar_month": stamp.strftime("%Y-%m")}
        if start + maximum_span > len(values):
            excluded.append({**block, "reason": "largest_schedule_does_not_fit_record"})
            continue
        current = []
        for spec in specs:
            e = data._sample_at_regular(values, mask, origin, start, TEMPORAL_RECORD_ID, spec, modes["embedding"], 792)
            r = data._sample_at_regular(values, mask, origin, start, TEMPORAL_RECORD_ID, spec, modes["raster"], 792)
            if e is None or r is None:
                current = []
                break
            data._weather_point_delete(e, 0.5, 99173)
            e["x_mark"] = torch.stack((e["x_mark"], r["x_mark"]), dim=-1)
            current.append(e)
        if len(current) != 4:
            excluded.append({**block, "reason": "one_or_more_schedules_have_no_observed_history_or_target"})
            continue
        block.update(anchor_digest=current[0]["anchor_digest"], source_block_digest=current[0]["source_block_digest"])
        blocks.append(block)
        samples.extend(current)
    dataset_windows = data.WindowDataset(samples, [spec.name for spec in specs])
    proof = {
        "rule": "2022 temporal rule: one block-origin anchor per 792-index block, all four schedules or none, no cap",
        "record_id": TEMPORAL_RECORD_ID,
        "qualifying_anchor_count": len(blocks),
        "realized_anchor_count": len(blocks),
        "partition_counts": {"score": dataset_windows.counts},
        "anonymous_roster_digest": {"score": data._roster_digest([b["anchor_digest"] for b in blocks])},
        "source_block_digest": {"score": data._roster_digest([b["source_block_digest"] for b in blocks])},
        "source_block_count": {"score": len(blocks)},
        "excluded_block_count": len(excluded),
        "excluded_blocks": excluded,
        "calendar_month_block_counts": dict(sorted(Counter(b["calendar_month"] for b in blocks).items())),
        "fixed_tensor_shapes": {"history_points": 72, "horizon_points": 18},
        "scaler_refitted": False,
        "replacement_anchor_searched": False,
    }
    return SimpleNamespace(dev_score=dataset_windows, channels=21, proof=proof)


def build_roster(name: str, payload, arm: str, metadata=None):
    """Return a namespace with ``dev_score`` windows, ``channels`` and ``proof``."""
    require(arm in CONFIG["coordinate_arms"], "unknown coordinate arm")
    if name == "ushcn_test":
        return _capped(payload, "ushcn", ROSTER_CAPS[name], arm)
    if name == "weather2020_test":
        require(payload[0] == WEATHER_TEST_RECORD_ID, "unexpected Weather test record id")
        return _capped(payload, "weather_jena2020", ROSTER_CAPS[name], arm)
    if name == "weather2023":
        require(metadata is not None, "temporal roster needs its preparation metadata")
        return _temporal(payload, metadata, arm)
    raise ValueError(f"unknown confirmation roster {name!r}")


def paired_across_arms(bundles) -> None:
    """All four arms must share windows, values, masks and proofs; only time labels differ."""
    reference = bundles["LL"]
    for bundle in bundles.values():
        require(bundle.proof == reference.proof, "roster proof changed across arms")
        _cross(reference.dev_score.samples, bundle.dev_score.samples)
